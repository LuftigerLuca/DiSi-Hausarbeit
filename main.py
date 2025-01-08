from bot_controller import BotController
from gesture_recognition import GestureRecognition

'''main funktion um die gesten zu erkennen. mit dobot ansteuerung'''
def main():
    base_path = "datasets"

    gesture_recognition = GestureRecognition(base_path)
    bot_controller = BotController()

    while True:
        file_path = input("Dateinamen angeben (z.B. dreieck5sec.csv) oder 'q' für quit: ")
        if file_path.lower() == "q":
            break
        execute_command(
            gesture_recognition.predict_new_gesture(file_path), bot_controller
        )


def execute_command(gesture, bot_controller):
    match gesture:
        case "shake":
            bot_controller.homing()
        case "circle":
            bot_controller.draw_circle()
        case "triangle":
            bot_controller.draw_triangle()
        case "square":
            bot_controller.draw_square()
        case _:
            print("Keine Geste gefunden")


if __name__ == "__main__":
    main()
