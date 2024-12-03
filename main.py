import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import tkinter as tk
from tkinter import simpledialog, messagebox
import face_recognition
from PIL import Image, ImageTk
from datetime import datetime, timedelta
from exif import Image as ExifImage

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests

class GooglePhotosLoader:
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()

    def _authenticate(self):
        CLIENT_SECRET_FILE = os.path.expanduser('credentials.json')
        
        SCOPES = [
            'https://www.googleapis.com/auth/photoslibrary.readonly',
            'https://www.googleapis.com/auth/photoslibrary.sharing'
        ]
        
        TOKEN_FILE = os.path.expanduser('token.json')

        if os.path.exists(TOKEN_FILE):
            self.creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_FILE, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open(TOKEN_FILE, 'w') as token:
                token.write(self.creds.to_json())
        
        self.service = build('photoslibrary', 'v1', credentials=self.creds, discoveryServiceUrl='https://photoslibrary.googleapis.com/$discovery/rest?version=v1')

    def list_photos(self, page_size=10):  # 기본값을 10으로 변경
        try:
            page_token = None
            photos = []
            
            while True:
                results = self.service.mediaItems().list(
                    pageSize=page_size,
                    pageToken=page_token
                ).execute()
                
                media_items = results.get('mediaItems', [])
                
                for item in media_items:
                    if item['mimeType'].startswith('image/'):
                        photo_info = {
                            'id': item['id'],
                            'filename': item['filename'],
                            'base_url': item['baseUrl'],
                            'mime_type': item['mimeType'],
                            'creation_time': item.get('mediaMetadata', {}).get('creationTime'),
                            'width': item.get('mediaMetadata', {}).get('width'),
                            'height': item.get('mediaMetadata', {}).get('height')
                        }
                        photos.append(photo_info)
                
                page_token = results.get('nextPageToken')
                if not page_token or len(photos) >= page_size:
                    break
            
            return photos
        
        except HttpError as error:
            print(f'Google Photos API 오류: {error}')
            return []
        except Exception as e:
            print(f'예상치 못한 오류: {e}')
            return []

    def download_photo(self, photo_info, download_path='./downloaded_photos'):
        os.makedirs(download_path, exist_ok=True)
        
        download_url = f"{photo_info['base_url']}=d"
        
        try:
            response = requests.get(download_url, 
                                    headers={'Authorization': f'Bearer {self.creds.token}'})
            
            file_path = os.path.join(download_path, photo_info['filename'])
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return file_path
        
        except Exception as e:
            print(f"다운로드 중 오류 발생: {e}")
            return None

class PhotoRelationshipExplorer:
    def __init__(self, download_path='./downloaded_photos', max_photos = None, load_from_google_photos=False):
        self.photos_path = download_path
        self.graph = nx.Graph()
        self.face_encodings = {}
        self.current_photo_index = {}
        self.photo_nodes = {}  # 여기에 파일 이름과 경로 매핑 추가
        self.dfs_path = []
        self.current_dfs_index = 0
        
        if load_from_google_photos:
            self._load_from_google_photos(max_photos)
        else:
            self.photos = self._load_photos()

        # 파일 이름과 파일 경로를 매핑
        self.photo_nodes = {photo['filename']: photo['filepath'] for photo in self.photos}

    def _load_from_google_photos(self, max_photos = None):
        google_loader = GooglePhotosLoader()
        photos = google_loader.list_photos(max_photos)
        
        self.photos = []
        for photo in photos:
            downloaded_path = google_loader.download_photo(photo)
            if downloaded_path:
                photo_info = {
                    'filepath': downloaded_path,
                    'filename': photo['filename'],
                    'metadata': self._extract_metadata(downloaded_path)
                }
                self.photos.append(photo_info)

    def _load_photos(self, max_photos=None):
        photos = []
        for filename in os.listdir(self.photos_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                filepath = os.path.join(self.photos_path, filename)
                photos.append({
                    'filepath': filepath,
                    'filename': filename,
                    'metadata': self._extract_metadata(filepath)
                })
                
                # 최대 이미지 개수에 도달하면 중단
                if max_photos is not None and len(photos) >= max_photos:
                    break
        
        return photos

    def _extract_metadata(self, filepath):
        metadata = {}
        try:
            with open(filepath, 'rb') as img_file:
                exif = ExifImage(img_file)
                
                if hasattr(exif, 'datetime'):
                    metadata['datetime'] = datetime.strptime(exif.datetime, '%Y:%m:%d %H:%M:%S')
                
                if hasattr(exif, 'gps_latitude') and hasattr(exif, 'gps_longitude'):
                    metadata['latitude'] = exif.gps_latitude
                    metadata['longitude'] = exif.gps_longitude
        except Exception as e:
            print(f"메타데이터 추출 오류: {e}")
        
        return metadata

    def _safe_read_image(self, filepath):
        image = cv2.imread(filepath)
        
        if image is None:
            print(f"Could not read image: {filepath}")
            return None
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect_faces(self):
        for photo in self.photos:
            try:
                image = self._safe_read_image(photo['filepath'])
                
                if image is None:
                    continue
                
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                self.face_encodings[photo['filename']] = face_encodings
            
            except Exception as e:
                print(f"Face detection warning for {photo['filename']}: {e}")
                self.face_encodings[photo['filename']] = []

    def build_relationship_graph(self, time_threshold=timedelta(days=1), location_threshold=0.01):
        for i, photo1 in enumerate(self.photos):
            for photo2 in self.photos[i+1:]:
                if 'datetime' in photo1['metadata'] and 'datetime' in photo2['metadata']:
                    time_diff = abs(photo1['metadata']['datetime'] - photo2['metadata']['datetime'])
                    if time_diff <= time_threshold:
                        self.graph.add_edge(photo1['filename'], photo2['filename'], type='time')
                
                if ('latitude' in photo1['metadata'] and 'latitude' in photo2['metadata']):
                    lat1, lon1 = photo1['metadata']['latitude'], photo1['metadata']['longitude']
                    lat2, lon2 = photo2['metadata']['latitude'], photo2['metadata']['longitude']
                    
                    distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                    if distance <= location_threshold:
                        self.graph.add_edge(photo1['filename'], photo2['filename'], type='location')

    def find_similar_faces(self, tolerance=0.6):
        for i, (photo1_name, encodings1) in enumerate(self.face_encodings.items()):
            for photo2_name, encodings2 in list(self.face_encodings.items())[i+1:]:
                for enc1 in encodings1:
                    matches = face_recognition.compare_faces(encodings2, enc1, tolerance=tolerance)
                    if any(matches):
                        self.graph.add_edge(photo1_name, photo2_name, type='face')

    def visualize_interactive_graph(self):
        root = tk.Tk()
        root.title("사진 관계 그래프")

        fig, ax = plt.subplots(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, k=0.5)  # Adjust layout for better spacing

        # Prepare node thumbnails with Gaussian blur
        node_images = {}
        for node in self.graph.nodes():
            try:
                img_path = os.path.join(self.photos_path, node)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize while keeping the original aspect ratio
                max_dim = 150
                h, w = img.shape[:2]
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(img_resized, (15, 15), 0)
                node_images[node] = blurred
            except Exception as e:
                print(f"Error processing image for node {node}: {e}")

        # Draw the graph
        nx.draw(self.graph, pos, ax=ax, with_labels=False, 
                edge_color='red',  # Red edges
                width=2,           # Slightly thicker edges
                node_color='lightblue', 
                node_size=300, 
                alpha=0.7)

        # Add image thumbnails and labels with white background
        for node, (x, y) in pos.items():
            if node in node_images:
                img = node_images[node]
                im = OffsetImage(img, zoom=1)  # Keep the zoom factor as 1
                im.image.axes = ax
                ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                ax.add_artist(ab)

            # Add labels with white background
            label_bg = plt.Rectangle((x - 0.02, y - 0.03), 0.04, 0.02, color='white', alpha=0.8)
            ax.add_patch(label_bg)
            ax.text(x, y - 0.03, node, fontsize=8, ha='center', va='center', zorder=10)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # DFS 경로 계산 메서드
        def calculate_dfs_path(start_node):
            # 노드 간 거리 계산 함수
            def node_distance(node1, node2):
                if 'latitude' in self.photos[self.photo_index[node1]]['metadata'] and \
                'latitude' in self.photos[self.photo_index[node2]]['metadata']:
                    lat1 = self.photos[self.photo_index[node1]]['metadata']['latitude']
                    lon1 = self.photos[self.photo_index[node1]]['metadata']['longitude']
                    lat2 = self.photos[self.photo_index[node2]]['metadata']['latitude']
                    lon2 = self.photos[self.photo_index[node2]]['metadata']['longitude']
                    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                return float('inf')

            self.photo_index = {photo['filename']: idx for idx, photo in enumerate(self.photos)}

            def custom_dfs(graph, start, visited=None):
                if visited is None:
                    visited = set()
                
                path = [start]
                visited.add(start)
                
                neighbors = list(graph.neighbors(start))
                neighbors.sort(key=lambda x: node_distance(start, x))
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        path.extend(custom_dfs(graph, neighbor, visited))
                
                return path

            # DFS 시작 노드 설정
            if start_node is None:
                start_node = max(self.graph.nodes(), key=lambda n: nx.degree(self.graph, n))

            # 경로 계산 및 초기화
            self.dfs_path = custom_dfs(self.graph, start_node)
            self.current_dfs_index = 0

        def dfs_next_image(event=None):
            if not self.dfs_path:
                calculate_dfs_path(list(self.graph.nodes())[0])

            if self.dfs_path:
                self.current_dfs_index = (self.current_dfs_index + 1) % len(self.dfs_path)
                current_node = self.dfs_path[self.current_dfs_index]
                image_path = self.photo_nodes.get(current_node)
                if image_path:
                    self.show_related_images([image_path], root)

        def dfs_prev_image(event=None):
            if not self.dfs_path:
                calculate_dfs_path(list(self.graph.nodes())[0])

            if self.dfs_path:
                self.current_dfs_index = (self.current_dfs_index - 1) % len(self.dfs_path)
                current_node = self.dfs_path[self.current_dfs_index]
                image_path = self.photo_nodes.get(current_node)
                if image_path:
                    self.show_related_images([image_path], root)

        # 클릭 이벤트 개선
        def on_node_click(event):
            if event.xdata is None or event.ydata is None:
                return

            clicked_node = None
            min_dist = float('inf')
            
            for node, (x, y) in pos.items():
                dist = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    clicked_node = node

            if clicked_node and min_dist < 150:  # 반경 조정
                print(f"Clicked on node: {clicked_node}")
                calculate_dfs_path(clicked_node)  # DFS 경로 갱신
                print("DFS Calc Complete!")

                # 클릭된 노드와 연결된 모든 이미지 경로 찾기
                related_nodes = list(self.graph.neighbors(clicked_node)) + [clicked_node]
                related_image_paths = [self.photo_nodes.get(node) for node in related_nodes if self.photo_nodes.get(node)]
                
                if related_image_paths:
                    self.show_related_images(related_image_paths, root)


        canvas.mpl_connect('button_press_event', on_node_click)
        canvas.mpl_connect('key_press_event', dfs_next_image) 

        # DFS 탐색 키 바인딩
        root.bind('<Right>', dfs_next_image)
        root.bind('<Left>', dfs_prev_image)

        root.mainloop()


    def show_related_images(self, image_paths, parent_window):
        # 기존에 열려있는 이미지 창 닫기
        if hasattr(self, 'image_window') and self.image_window:
            self.image_window.destroy()

        screen_width = parent_window.winfo_screenwidth()
        screen_height = parent_window.winfo_screenheight()

        # 새 창 생성
        self.image_window = tk.Toplevel(parent_window)
        self.image_window.title("관련 이미지")

        # 전체 이미지 경로 저장
        self.current_images = image_paths
        self.current_photo_index = 0

        image_label = tk.Label(self.image_window)
        image_label.pack()

        def show_image(index):
            image_path = self.current_images[index]

            # 이미지 로드
            image = self._safe_read_image(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return

            filename = os.path.basename(image_path)
            
            # 이미지 전처리 및 얼굴 인식
            face_image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(face_image)

            # 디버깅: 찾은 얼굴 수 출력
            print(f"Faces found: {len(face_locations)}")

            # 원본 이미지에 얼굴 위치에 사각형 그리기
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

            # OpenCV 이미지를 PIL 이미지로 변환
            pil_image = Image.fromarray(image)

            # 크기 조정: 화면 크기의 3/4로 설정
            pil_image.thumbnail((int(screen_width * 0.75), int(screen_height * 0.75)), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)

            # 메인 이미지 표시
            image_label.config(image=tk_image)
            image_label.image = tk_image
            self.image_window.title(f"Image {index+1}: {os.path.basename(image_path)}")

            # 기존 얼굴 썸네일 제거
            for widget in faces_frame.winfo_children():
                widget.destroy()

            # 하단 얼굴 썸네일 영역 생성
            for (top, right, bottom, left) in face_locations:
                try:
                    # 얼굴 영역 잘라내기 (원본 이미지에서)
                    face_image = face_image[top:bottom, left:right]
                    
                    # PIL 이미지로 변환
                    face_pil = Image.fromarray(face_image)
                    face_pil.thumbnail((100, 100), Image.LANCZOS)

                    # 얼굴 썸네일 표시
                    face_tk = ImageTk.PhotoImage(face_pil)
                    face_label = tk.Label(faces_frame, image=face_tk)
                    face_label.image = face_tk  # 참조 유지
                    face_label.pack(side=tk.LEFT, padx=5)
                except Exception as e:
                    print(f"Error creating face thumbnail: {e}")
                    print(f"Face location: top={top}, right={right}, bottom={bottom}, left={left}")
                    print(f"Image shape: {face_image.shape}")

        # 하단 얼굴 썸네일 영역 생성
        faces_frame = tk.Frame(self.image_window)
        faces_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # 키보드 이벤트로 이전/다음 이미지 탐색 가능
        def next_image(event=None):
            self.current_photo_index = (self.current_photo_index + 1) % len(self.current_images)
            show_image(self.current_photo_index)

        def prev_image(event=None):
            self.current_photo_index = (self.current_photo_index - 1) % len(self.current_images)
            show_image(self.current_photo_index)

        self.image_window.bind('<Right>', next_image)
        self.image_window.bind('<Left>', prev_image)

        # 초기 이미지 표시
        if self.current_images:
            show_image(0)

if __name__ == "__main__":
    # Create root window for input dialogs
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask user about loading from Google Photos
    use_google_photos = messagebox.askyesno(
        "Photo Source", 
        "Do you want to load photos from Google Photos?\n"
        "(If No, photos will be loaded from the local './downloaded_photos' directory)"
    )

    # Ask user for number of photos
    while True:
        try:
            max_photos = simpledialog.askinteger(
                "Photo Limit", 
                "How many photos do you want to load?\n"
                "(Enter a number between 1 and 100, or Cancel for all photos)",
                minvalue=1, 
                maxvalue=100
            )
            
            # If user cancels, set to None (load all photos)
            if max_photos is None:
                max_photos = None
            
            break
        except Exception as e:
            messagebox.showerror("Invalid Input", "Please enter a valid number of photos.")

    # Close the hidden root window
    root.destroy()

    # Create the PhotoRelationshipExplorer with user-specified parameters
    explorer = PhotoRelationshipExplorer(
        max_photos=max_photos, 
        load_from_google_photos=use_google_photos
    )

    # Set up Korean font for matplotlib
    rc('font', family='Malgun Gothic')  # Windows 사용 시
    plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 설정 시 마이너스 깨짐 방지

    # Build relationship graph
    explorer.build_relationship_graph()
    explorer.detect_faces()
    explorer.find_similar_faces()
    explorer.visualize_interactive_graph()