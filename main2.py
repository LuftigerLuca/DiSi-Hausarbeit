
from bot_controller import BotController
from gesture2 import GestureRecognition


def main():
    base_path = "datasets"
    gesture_recognition = GestureRecognition(base_path)

    #bot_controller = BotController()

    while True:
        file_path = input("Enter the file path for a new gesture or 'q' to quit: ")
        if file_path.lower() == 'q':
            break

        gesture = gesture_recognition.predict_new_gesture(file_path)
        print(f"Predicted gesture: {gesture}")



if __name__ == "__main__":
    main()
