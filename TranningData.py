class TrainingData():


    def __init__(self):
        self.trainingData = []
        self.trainingData.append({"class": "greeting", "sent": "how are you?"})
        self.trainingData.append({"class": "greeting", "sent": "how is your day?"})
        self.trainingData.append({"class": "greeting", "sent": "good day"})
        self.trainingData.append({"class": "greeting", "sent": "how is it going today?"})
        self.trainingData.append({"class": "goodbye", "sent": "have a nice day"})
        self.trainingData.append({"class": "goodbye", "sent": "see you later"})
        self.trainingData.append({"class": "goodbye", "sent": "have a nice day"})
        self.trainingData.append({"class": "goodbye", "sent": "talk to you soon"})
        self.trainingData.append({"class": "sandwich", "sent": "make me a sandwich"})
        self.trainingData.append({"class": "sandwich", "sent": "can you make a sandwich?"})
        self.trainingData.append({"class": "sandwich", "sent": "having a sandwich today?"})
        self.trainingData.append({"class": "sandwich", "sent": "what's for lunch?"})


    def getTrainData(self):
        return self.trainingData