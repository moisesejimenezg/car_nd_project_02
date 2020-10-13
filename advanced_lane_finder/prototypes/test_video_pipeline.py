from advanced_lane_finder.core.video_pipeline import VideoPipeline

pipeline = VideoPipeline(9, 6, 15)
pipeline.Prepare()

pipeline.Process(
    "./advanced_lane_finder/data/project_video.mp4",
    "./advanced_lane_finder/data/output_video/processed_project_video.mp4",
)

print("Offsets: " + str(pipeline.GetOffsets()))
print("Curvatures: " + str(pipeline.GetCurvatures()))
