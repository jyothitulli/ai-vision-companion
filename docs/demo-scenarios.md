# Demo scenarios

Use a real camera or still photos. Expected language is approximate; the system must not invent objects that YOLO/depth did not produce.

1. **Person approaching from left** — after two+ frames with growing left-side person box: "A person is approaching..."
2. **Chair in path** — center near chair: "A chair is approximately one to two meters ahead and may obstruct your path."
3. **Door** — only if a door-like specialized/open-vocab detection exists; otherwise an honest miss.
4. **Stairs** — pretrained COCO will often miss. Low-confidence specialized output must hedge. Fine-tune before claiming this demo.
5. **OCR** — photograph a receipt: READ, then "what is the total?"
6. **Find object** — "find my bottle" with a bottle in view.
7. **Continuous assistance** — same chair for many frames should not repeat the sentence every frame (cooldown).

Safety: a traffic light must produce crossing uncertainty language, never "cross now."
