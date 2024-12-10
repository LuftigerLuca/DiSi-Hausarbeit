from bot_controller import BotController

controller = BotController()

starting_point = controller.request_starting_point()


selection = input("Choose Action: \n 1 Homing \n 2 Circle \n 3 Triangle \n 4 Square \n")
match selection:
    case "1": controller.homing()
    case "2": controller.draw_circle(starting_point)
    case "3": controller.draw_triangle(starting_point)
    case "4": controller.draw_square(starting_point)