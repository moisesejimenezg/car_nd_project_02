from moviepy.editor import VideoFileClip

from pipeline import Pipeline


class VideoPipeline:
    def __init__(self, nx, ny, kernel_size=3):
        self.pipeline_ = Pipeline(nx, ny, kernel_size)

    def Prepare(self):
        self.pipeline_.Prepare()

    def GetCurvatures(self):
        self.pipeline_.GetCurvatures()

    def Process(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        processed_clip = clip.fl_image(self.pipeline_.Process)
        processed_clip.write_videofile(output_path, audio=False)
