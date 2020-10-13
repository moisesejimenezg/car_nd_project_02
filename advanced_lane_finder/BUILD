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
        "core/prev_poly.py",
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
    name = "test_calibration",
    srcs = ["prototypes/test_calibration.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
    ],
)

py_binary(
    name = "test_color",
    srcs = ["prototypes/test_color.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
    ],
)

py_binary(
    name = "test_gradient",
    srcs = ["prototypes/test_gradient.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
    ],
)

py_binary(
    name = "test_lines",
    srcs = ["prototypes/test_lines.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
    ],
)

py_binary(
    name = "test_perspective",
    srcs = ["prototypes/test_perspective.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
    ],
)

py_binary(
    name = "test_pipeline",
    srcs = ["prototypes/test_pipeline.py"],
    deps = [
        ":advanced_lane_finder",
    ],
    data = [
        ":data",
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
