from __future__ import annotations

TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
SPLIT = "split"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLIT_VALUES = (TRAIN, VALIDATION, TEST)

MIN_YEAR = 1922
MAX_YEAR = 2011

EXPECTED_INPUT_TRACKS = 1_000_000
EXPECTED_LABELED_TRACKS = 515_576
EXPECTED_UNLABELED_TRACKS = 484_424
EXPECTED_LABELED_ARTISTS = 28_223
EXPECTED_OFFICIAL_TRAIN_ARTISTS = 25_398
EXPECTED_OFFICIAL_TEST_ARTISTS = 2_822
EXPECTED_OFFICIAL_OMISSIONS = 3

EXPECTED_SPLITS = {
    TEST: {"artists": 2_822, "tracks": 49_436},
    TRAIN: {"artists": 22_867, "tracks": 420_013},
    VALIDATION: {"artists": 2_534, "tracks": 46_127},
}

OFFICIAL_SPLIT_COMMIT = "0c276e289606d5bd6f3991f713e7e9b1d4384e44"
OFFICIAL_TRAIN_SHA256 = "28c1cb3e50943e74a8a729b674f7e3240814fb1d2e07e0611179abd7d0e0057a"
OFFICIAL_TEST_SHA256 = "69655f5a21618dee29ca254ed75280f039ae127a8aac4170843de5b98b9b1dfd"

AUDIO_CONTRACT_VERSION = "shared_audio_628_v1"
AUDIO_FEATURE_COUNT = 628
AUDIO_FEATURE_ORDER_SHA256 = "70a34615d2a0c4734df885b08fcb752a83f4c757a1bd6339ad9cc6601aa5f0ec"

SCALAR_COLUMNS = (
    TRACK_ID,
    "loudness",
    "tempo",
    "duration",
    "key",
    "key_confidence",
    "mode",
    "mode_confidence",
    "time_signature",
    "time_signature_confidence",
    "end_of_fade_in",
    "start_of_fade_out",
    ARTIST_ID,
    "artist_name",
    "release",
    "release_7digitalid",
    "song_id",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
    "title",
    "track_7digitalid",
    YEAR,
)

SCALAR_TYPES = {
    **{
        column: "double"
        for column in (
            *SCALAR_COLUMNS[1:12],
            "song_hotttnesss",
            "artist_hotttnesss",
            "artist_familiarity",
        )
    },
    **{column: "string" for column in (TRACK_ID, ARTIST_ID, "artist_name", "release", "song_id", "title")},
    **{column: "int" for column in ("key", "mode", "time_signature", YEAR)},
    **{column: "bigint" for column in ("release_7digitalid", "track_7digitalid")},
}

LABEL_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, SPLIT)
ASSIGNMENT_COLUMNS = (TRACK_ID, ARTIST_ID, SPLIT)
LABEL_TYPES = {TRACK_ID: "string", ARTIST_ID: "string", YEAR: "int", SPLIT: "string"}
ASSIGNMENT_TYPES = {TRACK_ID: "string", ARTIST_ID: "string", SPLIT: "string"}
