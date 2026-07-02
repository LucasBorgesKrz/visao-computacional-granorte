"""
Detecção de caçamba cheia/vazia via análise de textura (Opção 2 - heurística)
Sem necessidade de treinar modelo - funciona direto no vídeo.
"""
import cv2
import numpy as np

# ── CONFIGURAÇÕES ────────────────────────────────────────────────────────────
VIDEO_ENTRADA = "WhatsApp_Video_2026-06-18_at_22_15_13.mp4"
VIDEO_SAIDA = "resultado_carga.mp4"

# ROI da caçamba (ajuste conforme sua câmera - calibrado para este vídeo)
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 100, 130, 420, 430

# Limite de densidade de bordas para considerar "cheio"
# Calibrado a partir da análise: vazio ~0.10, cheio ~0.20
LIMITE_CHEIO = 0.15

# ── PROCESSAMENTO ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_ENTRADA)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (w, h))

frame_idx = 0
status_anterior = None
eventos = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    cheio = edge_density >= LIMITE_CHEIO
    status = "CHEIO" if cheio else "VAZIO"

    # Registra mudança de status (para o alerta sonoro)
    if status != status_anterior:
        eventos.append((frame_idx, status))
        status_anterior = status

    # ── Desenha bounding box ──
    cor = (0, 0, 255) if cheio else (0, 255, 0)  # vermelho=cheio, verde=vazio
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), cor, 3)

    label = f"CACAMBA: {status} ({edge_density:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (ROI_X1, ROI_Y1 - th - 12), (ROI_X1 + tw + 10, ROI_Y1), cor, -1)
    cv2.putText(frame, label, (ROI_X1 + 5, ROI_Y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Processamento concluído: {frame_idx} frames")
print(f"Eventos de mudança de status:")
for ev in eventos:
    print(f"  Frame {ev[0]} (t={ev[0]/fps:.1f}s): {ev[1]}")
