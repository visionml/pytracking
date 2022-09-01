'''
The Ground Station represents a client in the communication with pytracking (server).

self.rtsp variable represents an rtsp stream, which should be set accordingly.

self.serverAddressPort represents server ID and port for communication.

Workflow of the Ground Station:

1. Tries to connect with the server, unitl it is connected.

2. After connection establishment, the ground station pop-up a window that shows continuous video feed. 
   Select a target object by drawing a bounding box using left mouse click.

3. The Bounding Box is sent to the server end to initialize the tracker.

4. The server after the initialization starts tracking the target object and sends back the tracked bounding box to Ground Station.

5. The Ground Station displays the countinuous receving tracked object.

Press Q for quit the Ground Station
Press R to re-select the object for tracking
Press T to Terminate both Ground Station and Server

'''

import cv2
from win32api import GetSystemMetrics
import socket
import threading
import time

class Tracking():

    def __init__(self):

        # Initializing Parameters
 
        self.rect = (0,0,0,0)
        self.startPoint = False
        self.endPoint = False
        self.Flag = False
        self.init_flag = False
        self.camera_flag = False
        self.rec_flag = False
        self.flag_recv = False
        self.flag_sending = False

        # RTSP Stream
        self.rtsp = 'rtsp://..'

        self.initBB = (0,0,0,0)
        self.initAA = (0,0,0,0)

        # Server ID and Port No
        self.serverAddressPort = ("127.0.0.1", 8554)
        self.bufferSize  = 1024


    def receive_msg(self, conn, bufferSize):

        bytesAddressPair = conn.recvfrom(bufferSize)

        message = b''
        message = bytesAddressPair[0]
        message = message.decode('utf-8')
        address = bytesAddressPair[1]

        return message, address

    def send_msg(self, conn, address, box):

        payload = str(box)
        payload = payload.encode('utf-8')
        conn.sendto(payload, address)
    
    def sending(self):
        counter = 0
        while self.flag_recv == False:
            Mess_conn = "Connection"
            self.send_msg(self.UDPClientSocket, self.serverAddressPort, Mess_conn)

            if self.flag_sending == True:
                break

            counter += 1
            if counter == 3:
                print('Trying to connect ...')
                counter = 0

            time.sleep(1)

    def closing_threads(self, cm, sd, con):

        self.thread_flag = True
        cm.join()

        self.flag_sending = True
        sd.join()

        self.thread_conn = True
        con.join()

    def connection(self):

        self.thread_conn = False

        msg_conn, addr = self.receive_msg(self.UDPClientSocket, self.bufferSize)
        if msg_conn == "Connected":
            print("Connection Establised.")

        self.flag_recv = True

        while True:
            if self.init_flag == True:
                if msg_conn == "Connected":

                    if self.init_flag == True:
                        self.send_msg(self.UDPClientSocket, self.serverAddressPort, self.initialization)
                    
                    if self.Flag == True:
                        Message2 = "Send Tracker Coordinates"
                        self.send_msg(self.UDPClientSocket, self.serverAddressPort, Message2)

                        msg, addr = self.receive_msg(self.UDPClientSocket, self.bufferSize)

                        if self.thread_conn == True:
                            break

                        if msg == "Tracker Lost":
                            print('Tracker Lost the Object')
                            cv2.destroyAllWindows()
                            break

                        else:
                            self.rec_flag = True
                            Trackerbox = (float(msg.split(",")[0][1:]), float(msg.split(",")[1][1:]), float(msg.split(",")[2][1:]), float(msg.split(",")[3][1:(len(msg.split(",")[3])-1)]))

                            self.Trackerbox = ((1/self.ratio)*Trackerbox[0], (1/self.ratio)*Trackerbox[1], (1/self.ratio)*Trackerbox[2], (1/self.ratio)*Trackerbox[3])            
            
            if self.thread_conn == True:
                break

            time.sleep(0.005)


    
    def camera(self):

        while True:
            cap = cv2.VideoCapture(self.rtsp)
            self.thread_flag = False

            while cap.isOpened() == True:
                self.success, self.frame = cap.read()
                self.camera_flag = True

                if self.success == False:
                    break

                if self.thread_flag == True:
                    break

                time.sleep(0.005)

            if self.thread_flag == True:
                break

            time.sleep(0.005)
        
    def main(self):

        while True:

            try:
                self.UDPClientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

                cam = threading.Thread(target=self.camera)
                cam.start()
                print('Camera thread Initiated')

                while True:
                    if self.camera_flag == True:
                        break

                snd = threading.Thread(target=self.sending)
                snd.start()
                print('Sending thread Initiated')

                conn = threading.Thread(target=self.connection)
                conn.start()
                print('Connection thread Initiated')
                
                while True:
                    if self.camera_flag == True:
                        frame_disp = self.frame.copy()

                        self.img_input, self.ratio = self.Img2ScreenRatio(frame_disp)

                        cv2.namedWindow("image")
                        cv2.setMouseCallback("image", self.GetCoordinates)

                        if self.startPoint == True and self.Flag == False:
                            cv2.rectangle(self.img_input, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (0, 255, 0), 2)
                                                
                        x = int((int(self.screen_width) - int(self.img_input.shape[1]))/2)

                        if self.endPoint == True:
                            Rectangle = self.RectPositioning(self.rect[0], self.rect[1], self.rect[2], self.rect[3])

                            if Rectangle[0] > 0 and Rectangle[1] > 0 and Rectangle[2] > 0 and Rectangle[3] > 0:
                                Rectangle = (int(self.ratio*Rectangle[0]), int(self.ratio*Rectangle[1]), int(self.ratio*Rectangle[2]), int(self.ratio*Rectangle[3]))

                                self.initBB = Rectangle
                                self.initialization = (self.initBB[0], self.initBB[1], (self.initBB[2] - self.initBB[0]), (self.initBB[3] - self.initBB[1]))

                                if self.initialization[2] >= 5 and self.initialization[3] >=5:
                                    self.Verify()
                                    self.Flag = True
                                
                                else:
                                    print('Select a Bounding Box instead of a Click')
                                    self.rect = (0,0,0,0)
                                    self.startPoint = False
                                    self.endPoint = False
                                    self.initBB = (0,0,0,0)
                                    self.initAA = (0,0,0,0)
                                    self.initialization = (0,0,0,0)
                                    self.Flag = False
                        
                        if self.rec_flag == True:
                            cv2.rectangle(self.img_input, (int(self.Trackerbox[0]), int(self.Trackerbox[1])), (int(self.Trackerbox[0]+self.Trackerbox[2]), int(self.Trackerbox[1]+self.Trackerbox[3])), (0, 0, 255), 2)
                        cv2.moveWindow("image",x,0)    
                        cv2.imshow('image', self.img_input)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Quit program using keyboard Interrupt")
                            Message3 = "Close"
                            self.send_msg(self.UDPClientSocket, self.serverAddressPort, Message3)

                            self.closing_threads(cam, snd, conn)

                            return

                        if cv2.waitKey(1) & 0xFF == ord('t'):
                            print("Terminate both GCS and Air Station")
                            Message3 = "Terminate"
                            self.send_msg(self.UDPClientSocket, self.serverAddressPort, Message3)

                            self.closing_threads(cam, snd, conn)

                            return

                        if cv2.waitKey(1) & 0xFF == ord('r'):
                            print("Reset program using keyboard Interrupt")
                            Message3 = "Close"
                            self.send_msg(self.UDPClientSocket, self.serverAddressPort, Message3)

                            self.thread_conn = True
                            conn.join()

                            self.rect = (0,0,0,0)
                            self.startPoint = False
                            self.endPoint = False
                            self.initBB = (0,0,0,0)
                            self.initAA = (0,0,0,0)
                            self.Trackerbox = (0,0,0,0)
                            self.flag_recv = False
                            self.initialization = (0,0,0,0)
                            self.init_flag = False
                            self.Flag = False

                            break

                        self.camera_flag = False
                    
                    time.sleep(0.005)

                cv2.destroyAllWindows()
                self.UDPClientSocket.close()

                time.sleep(0.005)

            except KeyboardInterrupt:
                print("Closed program using keyboard Interrupt")
                Message3 = "Close"
                self.send_msg(self.UDPClientSocket, self.serverAddressPort, Message3)

                self.closing_threads(cam, snd, conn)

                cv2.destroyAllWindows()
                self.UDPClientSocket.close()
    
    def Img2ScreenRatio(self, frame):

        image_height, image_width = frame.shape[:2]

        self.screen_width = GetSystemMetrics(0) # 1920 
        screen_height = GetSystemMetrics(1) #1080

        image_ratio = image_width / image_height
        screen_ratio = self.screen_width / screen_height

        if image_ratio < screen_ratio:
            img_input = cv2.resize(frame, (int(screen_height*image_ratio),screen_height))
            ratio = image_height / screen_height
        
        else:
            img_input = cv2.resize(frame, (self.screen_width,int(self.screen_width/image_ratio)))
            ratio = image_width / self.screen_width
        
        return img_input, ratio

    def GetCoordinates(self,event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect = (x, y, 0, 0)
            self.startPoint = True
            self.endPoint = False
            self.rect = (self.rect[0], self.rect[1], x, y)

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.endPoint == False:
                self.rect = (self.rect[0], self.rect[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.endPoint = True
    
    def RectPositioning(self,x1,y1,x2,y2):

        if x1 <= x2 and y1 <= y2:
            rect = (x1, y1, x2, y2)
            return rect
        
        elif x1 >= x2 and y1 >= y2:
            rect = (x2, y2, x1, y1)
            return rect
        
        elif x1 <= x2 and y1 >= y2:
            rect = (x1, y2, x2, y1)
            return rect

        elif x1 >= x2 and y1 <= y2:
            rect = (x2, y1, x1, y2)
            return rect
                
    
    def Verify(self):
        if self.initAA[0] != self.initBB[0] or self.initAA[1] != self.initBB[1] or self.initAA[2] != self.initBB[2] or self.initAA[3] != self.initBB[3]:
            print("Bounding Box Selected from GCS: ", self.initialization)
            self.init_flag = True
        
        self.initAA = self.initBB


Tracking().main()