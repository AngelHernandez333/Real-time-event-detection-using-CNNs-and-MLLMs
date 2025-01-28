from abc import ABC, abstractmethod
def DecisionMakerPerEvent():
    @abstractmethod
    def detections_treatment(self):
        pass
    @abstractmethod
    def decision_maker(self):
        pass
    @staticmethod
    def identify_persons():
        pass

def EventBicycle(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=["person", "bicycle"]
    def detections_treatment(self):
        pass
    def decision_maker(self):
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            correct = []
            for detection in detections:
                if detection[0] in classes_of_interest and detection[1] > 0.5:
                    correct.append(detection)
            decision = boxes_touching(correct)
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text
        else:
            return False, ""
    
if __name__ == "__main__":
    pass