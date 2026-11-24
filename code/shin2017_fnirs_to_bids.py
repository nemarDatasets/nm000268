"""Add the Shin et al. 2017 fNIRS recordings to the NEMAR BIDS trees nm000267 / nm000268.

The trees were exported by moabb 1.5.0 (Shin2017A / Shin2017B) and hold only EEG.
This script reads the upstream BBCI NIRS files (cnt.mat, mrk.mat, mnt.mat per
subject, from http://doc.ml.tu-berlin.de/hBCI/NIRS/NIRS_01-29.zip), writes one
SNIRF file per session next to the EEG file of the same session, and writes the
BIDS sidecars. It never writes inside an ``eeg/`` folder.

Two steps:

    python shin2017_fnirs_to_bids.py subject --nirs-src NIRS --bids-root ROOT --subject 1
    python shin2017_fnirs_to_bids.py dataset --nirs-src NIRS --archive ZIP --bids-root ROOT

``subject`` converts the six sessions of one subject. ``dataset`` copies the
source .mat files to sourcedata/NIRS and updates README.md and
dataset_description.json. ``--bids-root`` can be repeated.

Session mapping follows moabb (BaseShin2017._get_single_subject_data): entry
``ii`` of the cnt/mrk arrays is session ``{ii}imagery`` for ii in 0, 2, 4 and
``{ii}arithmetic`` for ii in 1, 3, 5. Every failed check raises.
"""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
from mne_nirs.io import write_raw_snirf
from scipy.io import loadmat

SCRIPT_NAME = "shin2017_fnirs_to_bids.py"
SCRIPT_VERSION = "1.0.0"
NIRS_URL = "http://doc.ml.tu-berlin.de/hBCI/NIRS/NIRS_01-29.zip"
PAPER_DOI = "10.1109/TNSRE.2016.2628057"
SUBJECTS = tuple(range(1, 30))
N_SESSIONS = 6
SOURCE_FILES = ("cnt.mat", "mrk.mat", "mnt.mat")
# cnt[ii].title for the two session kinds, and moabb's event names per BBCI class.
SESSION_TITLE = {"imagery": "MI", "arithmetic": "MA"}
TRIAL_TYPE = {"LMI": "left_hand", "RMI": "right_hand", "MA": "subtraction", "BL": "rest"}
WAVELENGTH_SUFFIX = ("lowWL", "highWL")
EXPECTED_CHANNELS = 72  # 36 source-detector pairs x 2 wavelengths
# mnt.pos_3d lies on the unit sphere; 0.095 m is MNE's default head radius.
HEAD_RADIUS_M = 0.095
# Every channel joins neighbouring 10-10 positions (about 27 to 37 mm at this radius).
MAX_SOURCE_DETECTOR_M = 0.045
# NIRS markers sit on an 80 ms grid (12.5 Hz), EEG onsets on a 5 ms grid.
MAX_TIMING_MISMATCH_S = 0.1
EEG_KEYS_COPIED = (
    "TaskName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "CapManufacturer",
    "CapManufacturersModelName",
)
SESSION_NIRS_META = {
    "imagery": {
        "TaskDescription": (
            "Left vs right hand motor imagery (kinesthetic imagery of opening and closing hands). "
            "Trial duration: 10.0s, cue: 2.0s."
        ),
        "Instructions": (
            "Perform kinesthetic motor imagery of opening and closing the left or right hand "
            "according to the visual arrow cue for 10 s, followed by a 15-17 s rest period."
        ),
        "CogAtlasID": "https://www.cognitiveatlas.org/task/id/trm_4c8a834779883/",
    },
    "arithmetic": {
        "TaskDescription": (
            "Mental arithmetic (serial subtraction of a one-digit number from a three-digit number) "
            "versus baseline/rest. Trial duration: 10.0s, cue: 2.0s."
        ),
        "Instructions": (
            "Perform serial subtraction of a one-digit number from a three-digit number during "
            "arithmetic trials, or relax during baseline/rest trials, for 10 s followed by a 15-17 s rest period."
        ),
    },
}
COORDSYSTEM = {
    "NIRSCoordinateSystem": "Other",
    "NIRSCoordinateUnits": "m",
    "NIRSCoordinateSystemDescription": (
        "Idealised spherical head model of the BBCI toolbox montage file "
        f"(mnt.mat, pos_3d on the unit sphere) scaled to a head radius of {HEAD_RADIUS_M} m. "
        "Origin at the centre of the sphere, +X towards the right ear (T8), "
        "+Y towards Fpz (anterior), +Z towards the vertex (Cz)."
    ),
    "NIRSCoordinateProcessingDescription": (
        "Nominal 10-10 cap positions, not digitised. The paper reports a nominal "
        "inter-optode distance of 30 mm."
    ),
}
README_HEADING = "fNIRS Acquisition\n-----------------\n"
README_ANCHOR = "Participants\n------------\n"
README_SECTION = README_HEADING + (
    f"  Added by code/{SCRIPT_NAME} from the upstream NIRS archive ({NIRS_URL}).\n"
    "  Device: NIRScout (NIRx GmbH, Berlin, Germany), continuous wave, wavelengths 760 and 850 nm\n"
    "  Optodes: 14 sources and 16 detectors on the EEG cap (frontal, motor and visual areas), "
    "36 source-detector channels (72 time series), nominal inter-optode distance 30 mm\n"
    "  Sampling rate: 10.0 Hz as distributed (the paper reports acquisition at 12.5 Hz)\n"
    "  Data: raw light intensities in volts, not processed, one SNIRF file per session in sub-*/ses-*/nirs/\n"
    f"  Optode positions: nominal 10-10 cap positions from the BBCI montage file, scaled to a {HEAD_RADIUS_M} m "
    "head radius; not digitised\n"
    "  Events: same trial order, trial_type and value as the EEG events of the same session. The fNIRS "
    "recording starts about 44 to 58 s before the EEG recording, so onsets are on the fNIRS clock; "
    "align the two recordings on their first event.\n"
    "  Source files: sourcedata/NIRS/subject NN/{cnt,mrk,mnt}.mat\n"
)
SOURCEDATA_HEADING = "## NIRS\n"
SOURCEDATA_SECTION = SOURCEDATA_HEADING + (
    f"\n`NIRS/subject NN/{{cnt,mrk,mnt}}.mat` are the fNIRS files of the upstream archive\n"
    f"<{NIRS_URL}>, byte-for-byte, under their upstream names. They were not retrieved through\n"
    f"MOABB (its loader does not read fNIRS) but by `code/{SCRIPT_NAME}`, which also converted them\n"
    "to the SNIRF files in `sub-*/ses-*/nirs/`. Per-file SHA-256 checksums are in\n"
    "`NIRS/sourcedata_provenance.json`. The artifact-experiment files of the archive\n"
    "(`*_artifact.mat`) are not included.\n"
)


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent="\t", ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_mat(path, key):
    return loadmat(path, squeeze_me=True, struct_as_record=False)[key]


def session_name(index):
    """moabb's session key for entry ``index`` of the cnt/mrk arrays."""
    return f"{index}imagery" if index % 2 == 0 else f"{index}arithmetic"


def read_optodes(mnt, kind):
    """Optodes of one kind used by a channel, renumbered 1..N in mnt order."""
    struct, column, prefix = (mnt.source, 0, "S") if kind == "source" else (mnt.detector, 1, "D")
    slots = sorted({int(slot) for slot in mnt.sd[:, column]})
    return [
        {
            "name": f"{prefix}{number}",
            "type": kind,
            "slot": slot,
            "label": str(struct.clab[slot - 1]),
            "position": struct.pos_3d[:, slot - 1] * HEAD_RADIUS_M,  # (3,) metres
        }
        for number, slot in enumerate(slots, start=1)
    ]


def read_channels(mnt, sources, detectors, wavelengths):
    """One entry per (pair, wavelength), pairs in mnt order, 760 nm before 850 nm."""
    source_of = {optode["slot"]: optode for optode in sources}
    detector_of = {optode["slot"]: optode for optode in detectors}
    n_pairs = len(mnt.sd)
    channels = []
    for pair, (source_slot, detector_slot) in enumerate(mnt.sd):
        source, detector = source_of[int(source_slot)], detector_of[int(detector_slot)]
        assert str(mnt.clab[pair]) == source["label"] + detector["label"], mnt.clab[pair]
        for band, wavelength in enumerate(wavelengths):
            channels.append(
                {
                    "name": f"{source['name']}_{detector['name']} {int(wavelength)}",
                    "source": source,
                    "detector": detector,
                    "wavelength": int(wavelength),
                    "bbci_label": f"{mnt.clab[pair]}{WAVELENGTH_SUFFIX[band]}",
                    "column": pair + band * n_pairs,  # column of cnt.x
                }
            )
    return channels


def session_events(mrk, eeg_events, sfreq):
    """NIRS events, checked against the EEG events of the same session."""
    trial_types = [TRIAL_TYPE[str(mrk.className[code - 1])] for code in mrk.event.desc]
    assert trial_types == [row["trial_type"] for row in eeg_events], "trial order differs from EEG"
    onsets = mrk.time / 1000.0 - 1.0 / sfreq  # BBCI mrk.time is 1-based (pos*1000/fs)
    eeg_onsets = np.array([float(row["onset"]) for row in eeg_events])
    mismatch = np.abs((onsets - onsets[0]) - (eeg_onsets - eeg_onsets[0])).max()
    assert mismatch <= MAX_TIMING_MISMATCH_S, f"trial timing differs from EEG by {mismatch:.3f} s"
    rows = [
        {
            "onset": round(float(onset), 3),
            "duration": row["duration"],
            "trial_type": row["trial_type"],
            "value": row["value"],
            "sample": int(round(float(onset) * sfreq)),
        }
        for onset, row in zip(onsets, eeg_events)
    ]
    return rows, float(onsets[0] - eeg_onsets[0]), float(mismatch)


def build_raw(cnt, channels, events, meas_date, subject):
    """MNE Raw of raw light intensities (volts) with optode positions and events."""
    assert [str(label) for label in cnt.clab] == [
        channel["bbci_label"] for channel in sorted(channels, key=lambda c: c["column"])
    ], "cnt.clab does not match mnt"
    info = mne.create_info([c["name"] for c in channels], float(cnt.fs), "fnirs_cw_amplitude")
    # Positions are written straight into loc. raw.set_montage is not used: with a
    # montage that has no fiducials it gave every optode the position of another one.
    for ch, channel in zip(info["chs"], channels):
        source, detector = channel["source"]["position"], channel["detector"]["position"]
        ch["loc"][:3] = (source + detector) / 2
        ch["loc"][3:6] = source
        ch["loc"][6:9] = detector
        ch["loc"][9] = channel["wavelength"]
    data = cnt.x[:, [c["column"] for c in channels]].T  # (n_channels, n_times)
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(meas_date)
    raw.info["subject_info"] = {"his_id": f"sub-{subject}"}
    raw.set_annotations(
        mne.Annotations(
            onset=[row["onset"] for row in events],
            duration=[float(row["duration"]) for row in events],
            description=[row["trial_type"] for row in events],
        )
    )
    return raw


def check_snirf(path, raw, channels, events):
    """Read the SNIRF file back and compare it with the source values."""
    back = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    assert len(back.ch_names) == EXPECTED_CHANNELS, len(back.ch_names)
    assert set(back.get_channel_types()) == {"fnirs_cw_amplitude"}
    assert back.info["sfreq"] == raw.info["sfreq"], back.info["sfreq"]
    order = [back.ch_names.index(name) for name in raw.ch_names]
    assert np.allclose(back.get_data()[order], raw.get_data(), rtol=1e-12, atol=0)
    expected = np.array(
        [[*c["source"]["position"], *c["detector"]["position"], c["wavelength"]] for c in channels]
    )  # (n_channels, 7), from mnt
    assert np.allclose(np.array([back.info["chs"][i]["loc"][3:10] for i in order]), expected, atol=1e-9)
    distances = np.linalg.norm(expected[:, :3] - expected[:, 3:6], axis=1)
    assert distances.max() <= MAX_SOURCE_DETECTOR_M, f"source-detector distance {distances.max():.3f} m"
    assert len(back.annotations) == len(events), len(back.annotations)
    by_onset = np.argsort(back.annotations.onset)
    assert list(back.annotations.description[by_onset]) == [row["trial_type"] for row in events]
    assert np.allclose(back.annotations.onset[by_onset], [row["onset"] for row in events], atol=1e-6)


def nirs_sidecar(eeg_sidecar, raw, n_sources, n_detectors, session_kind):
    copied = {key: eeg_sidecar[key] for key in EEG_KEYS_COPIED if key in eeg_sidecar}
    return {
        **copied,
        **SESSION_NIRS_META[session_kind],
        "Manufacturer": "NIRx",
        "ManufacturersModelName": "NIRScout",
        "SamplingFrequency": raw.info["sfreq"],
        "NIRSChannelCount": len(raw.ch_names),
        "NIRSSourceOptodeCount": n_sources,
        "NIRSDetectorOptodeCount": n_detectors,
        "ShortChannelCount": 0,
        "NIRSPlacementScheme": "10-10",
        "RecordingDuration": raw.n_times / raw.info["sfreq"],
        "RecordingType": "continuous",
        "SoftwareFilters": {
            "Resampling": {
                "Description": "The paper reports acquisition at 12.5 Hz; the distributed "
                "BBCI files (cnt.fs) are sampled at 10 Hz. No processing was added here."
            }
        },
        "SoftwareVersions": "n/a",
        "DeviceSerialNumber": "n/a",
        "SubjectArtefactDescription": "n/a",
    }


def channel_rows(channels, sfreq):
    return [
        {
            "name": channel["name"],
            "type": "NIRSCWAMPLITUDE",
            "source": channel["source"]["name"],
            "detector": channel["detector"]["name"],
            "wavelength_nominal": channel["wavelength"],
            "units": "V",
            "sampling_frequency": sfreq,
            "short_channel": "false",  # the validator counts this column against ShortChannelCount
            "status": "good",
            "description": f"{channel['source']['label']}-{channel['detector']['label']} "
            f"{channel['wavelength']} nm (BBCI label {channel['bbci_label']})",
        }
        for channel in channels
    ]


def optode_rows(optodes):
    return [
        {
            "name": optode["name"],
            "type": optode["type"],
            "x": round(float(optode["position"][0]), 6),
            "y": round(float(optode["position"][1]), 6),
            "z": round(float(optode["position"][2]), 6),
            "description": f"10-10 position {optode['label']} (index {optode['slot']} in mnt.{optode['type']})",
        }
        for optode in optodes
    ]


def convert_session(root, subject, index, cnt, mrk, channels, sources, detectors):
    """Write the nirs/ folder of one session of one BIDS tree."""
    session = session_name(index)
    session_dir = root / f"sub-{subject}" / f"ses-{session}"
    (eeg_file,) = sorted((session_dir / "eeg").glob("*_eeg.bdf"))
    stem = eeg_file.name.removesuffix("_eeg.bdf")  # sub-N_ses-S_task-T_run-R
    assert cnt.title == SESSION_TITLE[session.lstrip("0123456789")], (session, cnt.title)

    eeg_events = read_tsv(session_dir / "eeg" / f"{stem}_events.tsv")
    events, offset, mismatch = session_events(mrk, eeg_events, float(cnt.fs))
    (scan,) = [row for row in read_tsv(session_dir / f"sub-{subject}_ses-{session}_scans.tsv")
               if row["filename"] == f"eeg/{eeg_file.name}"]
    meas_date = datetime.fromisoformat(scan["acq_time"]).astimezone(timezone.utc)
    optodes = [*sources, *detectors]
    raw = build_raw(cnt, channels, events, meas_date, subject)

    nirs_dir = session_dir / "nirs"
    nirs_dir.mkdir(exist_ok=True)
    snirf = nirs_dir / f"{stem}_nirs.snirf"
    write_raw_snirf(raw, snirf)
    check_snirf(snirf, raw, channels, events)
    eeg_sidecar = json.loads((session_dir / "eeg" / f"{stem}_eeg.json").read_text(encoding="utf-8"))
    write_json(nirs_dir / f"{stem}_nirs.json", nirs_sidecar(eeg_sidecar, raw, len(sources), len(detectors), session.lstrip("0123456789")))
    write_tsv(nirs_dir / f"{stem}_channels.tsv", channel_rows(channels, raw.info["sfreq"]))
    write_tsv(nirs_dir / f"{stem}_events.tsv", events)
    shutil.copyfile(session_dir / "eeg" / f"{stem}_events.json", nirs_dir / f"{stem}_events.json")
    write_tsv(nirs_dir / f"sub-{subject}_ses-{session}_optodes.tsv", optode_rows(optodes))
    write_json(nirs_dir / f"sub-{subject}_ses-{session}_coordsystem.json", COORDSYSTEM)
    print(
        f"{root.name} sub-{subject} ses-{session}: {raw.n_times} samples at {raw.info['sfreq']} Hz, "
        f"{len(raw.ch_names)} channels, {len(events)} events, starts {offset:+.2f} s before EEG, "
        f"trial timing mismatch {mismatch:.3f} s -> {snirf.name}"
    )


def convert_subject(nirs_src, roots, subject):
    folder = nirs_src / f"subject {subject:02d}"
    cnt, mrk, mnt = (load_mat(folder / name, name.removesuffix(".mat")) for name in SOURCE_FILES)
    assert len(cnt) == len(mrk) == N_SESSIONS, (len(cnt), len(mrk))
    sources, detectors = read_optodes(mnt, "source"), read_optodes(mnt, "detector")
    assert (len(sources), len(detectors)) == (cnt[0].nSources, cnt[0].nDetectors)
    for index in range(N_SESSIONS):
        channels = read_channels(mnt, sources, detectors, cnt[index].wavelengths)
        assert len(channels) == EXPECTED_CHANNELS, len(channels)
        for root in roots:
            convert_session(root, subject, index, cnt[index], mrk[index], channels, sources, detectors)


def add_sourcedata(root, nirs_src, archive):
    """Copy the NIRS .mat files read by the converter, with checksums."""
    dest = root / "sourcedata" / "NIRS"
    files = []
    for subject in SUBJECTS:
        for name in SOURCE_FILES:
            member = Path(f"subject {subject:02d}") / name
            (dest / member).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(nirs_src / member, dest / member)
            files.append(
                {
                    "file": member.as_posix(),
                    "bytes": (dest / member).stat().st_size,
                    "sha256": sha256(dest / member),
                    "subject": str(subject),
                }
            )
    provenance = {
        "dataset": root.name,
        "modality": "nirs",
        "retrieved": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).date().isoformat(),
        "retrieved_by": f"{SCRIPT_NAME} {SCRIPT_VERSION}",
        "source_url": NIRS_URL,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "license": "GPL-3.0",
        "doi": PAPER_DOI,
        "naming": "upstream_archive_member_names",
        "not_included": "cnt_artifact.mat, mrk_artifact.mat, mnt_artifact.mat, subject 14/readme.txt",
        "n_files": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
        "n_subjects_recorded": len(SUBJECTS),
    }
    (dest / "sourcedata_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")


def add_text_section(path, heading, section, anchor=None):
    """Insert ``section`` before ``anchor`` (or append it) unless it is already there."""
    text = path.read_text(encoding="utf-8")
    if heading in text:
        return
    if anchor is None:
        path.write_text(text.rstrip("\n") + "\n\n" + section, encoding="utf-8")
        return
    assert text.count(anchor) == 1, f"{path}: expected one {anchor!r}"
    path.write_text(text.replace(anchor, section + "\n" + anchor), encoding="utf-8")


def add_generated_by(path):
    description = json.loads(path.read_text(encoding="utf-8"))
    if any(entry.get("Name") == SCRIPT_NAME for entry in description["GeneratedBy"]):
        return
    entry = {
        "Name": SCRIPT_NAME,
        "Version": SCRIPT_VERSION,
        "Description": "Converted the upstream BBCI NIRS .mat files to SNIRF and added the BIDS "
        f"nirs sidecars next to the EEG files (code/{SCRIPT_NAME}, mne {mne.__version__}).",
    }
    updated = {**description, "GeneratedBy": [*description["GeneratedBy"], entry]}
    path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finalize_dataset(root, nirs_src, archive):
    n_snirf = len(list(root.glob("sub-*/ses-*/nirs/*_nirs.snirf")))
    n_eeg = len(list(root.glob("sub-*/ses-*/eeg/*_eeg.bdf")))
    assert n_snirf == n_eeg, f"{root.name}: {n_snirf} SNIRF files for {n_eeg} EEG files"
    add_sourcedata(root, nirs_src, archive)
    add_text_section(root / "sourcedata" / "README.md", SOURCEDATA_HEADING, SOURCEDATA_SECTION)
    add_text_section(root / "README.md", README_HEADING, README_SECTION, anchor=README_ANCHOR)
    add_generated_by(root / "dataset_description.json")
    shutil.copyfile(Path(__file__), root / "code" / SCRIPT_NAME)
    print(f"{root.name}: {n_snirf} SNIRF files, sourcedata/NIRS, README.md and dataset_description.json updated")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("subject", "dataset"))
    parser.add_argument("--nirs-src", type=Path, required=True, help="extracted NIRS_01-29.zip")
    parser.add_argument("--bids-root", type=Path, required=True, action="append")
    parser.add_argument("--subject", type=int, choices=SUBJECTS, help="for the subject step")
    parser.add_argument("--archive", type=Path, help="NIRS_01-29.zip, for the dataset step")
    args = parser.parse_args()
    if args.step == "subject":
        convert_subject(args.nirs_src, args.bids_root, args.subject)
        return
    for root in args.bids_root:
        finalize_dataset(root, args.nirs_src, args.archive)


if __name__ == "__main__":
    main()
