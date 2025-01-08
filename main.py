from bot_controller import BotController
from gesture_recognition import GestureRecognition

homed = False

'''main funktion um die gesten zu erkennen. mit dobot ansteuerung
    testing = False: wenn True, wird der Bot nicht angesprochen
'''
def main(testing=False):
    base_path = "datasets"

    gesture_recognition = GestureRecognition(base_path)

    while True:
        file_path = input("Dateinamen angeben (z.B. dreieck5sec.csv) oder 'q' für quit: ")
        if file_path.lower() == "q":
            break
        gesture = gesture_recognition.predict_new_gesture(file_path)
        if not testing:
            bot_controller = BotController()
            execute_command(gesture, bot_controller)


def execute_command(gesture, bot_controller):
    global homed
    if not homed:
        bot_controller.homing()
        homed = True

    match gesture:
        case "circle":
            bot_controller.draw_circle()
        case "triangle":
            bot_controller.draw_triangle()
        case "square":
            bot_controller.draw_square()
        case _:
            print("Keine Geste gefunden")


if __name__ == "__main__":
    main(True)
