from engine.vision.camera import Camera

camera = Camera()

camera.open()

frame = camera.read()

print(frame)

camera.release()