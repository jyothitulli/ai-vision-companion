# Privacy

Default: **do not store camera frames**.

Flow: capture → RAM / request body → inference tensors → spoken JSON → release.

`STORE_IMAGES=false` in `.env.example`. There is no image blob column on `Interaction` or `DetectionEvent`.

If image logging is ever added it must be explicit opt-in, user-visible, and deletable. Do not log raw pixels in application logs (`LOG_IMAGE_METADATA_ONLY=true`).
