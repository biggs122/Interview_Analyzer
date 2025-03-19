from deepface import DeepFace

DeepFace.verify(
    "/Users/abderrahim_boussyf/interview_analyzer/data/facial/FER2013/test/angry/PrivateTest_88305.jpg",
    "/Users/abderrahim_boussyf/interview_analyzer/data/facial/FER2013/test/neutral/PrivateTest_687498.jpg",
    detector_backend='mtcnn'
)

