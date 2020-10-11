from video_pipeline import VideoPipeline

pipeline = VideoPipeline(9, 6, 15)
pipeline.Prepare()

pipeline.Process('../project_video.mp4', '../output_video/processed_project_video.mp4')
print(pipeline.GetCurvatures())
