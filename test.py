import torch, numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

model = InceptionResnetV1(pretrained='vggface2').eval()
tfm = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    (lambda x: (x - 0.5) / 0.5),  # זהה ל-fixed_image_standardization
])

def emb(img):
    with torch.no_grad():
        t = tfm(img).unsqueeze(0)
        e = model(t).float()
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        return e[0].cpu().numpy()

black = Image.fromarray(np.zeros((200,200,3), dtype=np.uint8))
white = Image.fromarray(np.ones((200,200,3), dtype=np.uint8)*255)
e_black = emb(black); e_white = emb(white)
print("cos(black, white) =", float(e_black @ e_white))

# נסה גם שתי תמונות פנים שונות (אחת שלך + אחת אחרת):
img1 = Image.open(r"query_images\RonaldoImage1.jpeg").convert("RGB")
img2 = Image.open(r"query_images\MessiImage1.jpg").convert("RGB")
e1 = emb(img1); e2 = emb(img2)
print("cos(img1, img2) =", float(e1 @ e2))
