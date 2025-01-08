from gesture_recognition import GestureRecognition

'''main funktion um die gesten zu erkennen. hier ohne den dobot, um nur die recognition zu testen'''
def main():
    base_path = "datasets"
    gesture_recognition = GestureRecognition(base_path)

    while True:
        file_path = input("Dateinamen angeben (z.B. dreieck5sec.csv) oder 'q' für quit: ")
        if file_path.lower() == 'q':
            break

        gesture = gesture_recognition.predict_new_gesture(file_path)
        print(f"Erkannte Geste: {gesture}")



if __name__ == "__main__":
    main()
