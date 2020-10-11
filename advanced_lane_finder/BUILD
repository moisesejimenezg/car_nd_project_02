load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:defs.bzl", "py_binary")
load("@deps//:requirements.bzl", "requirement")

filegroup(
    name = "data",
    srcs = glob([
        "data/**/*.*",
    ]),
)

py_library(
    name = "advanced_lane_finder",
    srcs = [
        "__init__.py",
        "core/calibration.py",
        "core/color.py",
        "core/geometry.py",
        "core/gradient.py",
        "core/lines.py",
        "core/perspective.py",
        "core/pipeline.py",
        "core/video_pipeline.py",
    ],
    deps = [
        requirement("matplotlib"),
        requirement("moviepy"),
        requirement("numpy"),
        requirement("opencv-contrib-python-headless"),
    ],
)

py_binary(
    name = "test_video_pipeline",
    srcs = ["prototypes/test_video_pipeline.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
    ],
)
