class Plate:
    plate_image=None
    def __init__(self,cropped_image_of_plate):
        self.plate_image = cropped_image_of_plate

class Char:
    x1=0
    y1=0
    x2=0
    y2=0
    char_image=None
    def __init__(self,x1,y1,x2,y2,cropped_image_of_char):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.char_image = cropped_image_of_char