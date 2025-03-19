import speechbrain
print(speechbrain.__version__)
import speechbrain.inference
print(dir(speechbrain.inference))
from speechbrain.inference import EncoderClassifier
print("EncoderClassifier imported successfully!")