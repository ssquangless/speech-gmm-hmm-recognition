from speech.hmm import load_bundle

b = load_bundle("weight/hmm_gmm_speech_models.pkl")
print("Temperature =", b["thresholds"]["temperature"])