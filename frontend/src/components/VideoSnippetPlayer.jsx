import { useEffect, useRef, useState } from "react";

/**
 * VideoSnippetPlayer
 *
 * Draws a moving face box overlay (one box per frame, best score above threshold)
 * on top of a snippet <video>. Boxes are fetched from /api/match_boxes.
 *
 * Props:
 *  - src: string
 *  - personId: string
 *  - videoName: string
 *  - startFrame: number
 *  - endFrame: number
 *  - fps: number
 *  - threshold?: number
 *  - style?: React.CSSProperties
 */
export default function VideoSnippetPlayer({
  src,
  personId,
  videoName,
  startFrame,
  endFrame,
  fps,
  threshold,
  style = {},
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // frame_idx -> { box:[x,y,w,h], score:number }
  const [boxesMap, setBoxesMap] = useState(new Map());
  const [metaSize, setMetaSize] = useState({ frame_w: null, frame_h: null, fps_meta: null });

  // Sticky state
  const lastBoxRef = useRef(null);   // { box:[x,y,w,h], score:number }
  const lastIdxRef = useRef(null);   // frame_idx

  // Fetch best-per-frame boxes for the snippet window
  useEffect(() => {
    let abort = false;

    async function fetchBoxes() {
      try {
        const params = new URLSearchParams({
          person_id: personId,
          video: videoName,
          start_idx: String(startFrame),
          end_idx: String(endFrame),
        });
        if (typeof threshold === "number") params.set("threshold", String(threshold));

        const res = await fetch(`http://localhost:5000/api/match_boxes?${params.toString()}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load boxes");
        if (abort) return;

        const map = new Map();
        for (const r of data.boxes || []) {
          map.set(Number(r.frame_idx), { box: r.box.map(Number), score: Number(r.score || 0) });
        }
        setBoxesMap(map);
        setMetaSize({
          frame_w: (data.frame_w != null && data.frame_h != null) ? Number(data.frame_w) : null,
          frame_h: (data.frame_w != null && data.frame_h != null) ? Number(data.frame_h) : null,
          fps_meta: (typeof data.fps === "number") ? data.fps : null,
        });

        // reset sticky state
        lastBoxRef.current = null;
        lastIdxRef.current = null;
      } catch (e) {
        console.warn("[VideoSnippetPlayer] overlay fetch failed:", e);
        setBoxesMap(new Map());
        setMetaSize({ frame_w: null, frame_h: null, fps_meta: null });
        lastBoxRef.current = null;
        lastIdxRef.current = null;
      }
    }

    fetchBoxes();
    return () => { abort = true; };
  }, [personId, videoName, startFrame, endFrame, threshold]);

  // Resize canvas to match displayed <video> rect
  function resizeCanvasToVideo() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const rect = video.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
  }

  // Compute scale + offset to account for letterbox ("contain" behavior)
  function computeLetterboxTransform() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return { sx: 1, sy: 1, ox: 0, oy: 0 };

    const srcW = metaSize.frame_w || video.videoWidth || canvas.width;
    const srcH = metaSize.frame_h || video.videoHeight || canvas.height;
    const dstW = canvas.width;
    const dstH = canvas.height;

    const scale = Math.min(dstW / srcW, dstH / srcH);
    const drawnW = srcW * scale;
    const drawnH = srcH * scale;
    const ox = (dstW - drawnW) / 2;
    const oy = (dstH - drawnH) / 2;

    return { sx: scale, sy: scale, ox, oy };
  }

  // Draw one frame's overlay (with sticky box)
  function drawOverlay() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // current index in original video
    const currentTime = video.currentTime || 0;
    const curOffsetFrames = Math.floor(currentTime * fps);
    const curFrameIdx = startFrame + curOffsetFrames;

    // box for current frame
    const found = boxesMap.get(curFrameIdx);

    // Sticky: if none for this frame, use last
    let toDraw = null;
    if (found) {
      lastBoxRef.current = found;
      lastIdxRef.current = curFrameIdx;
      toDraw = found;
    } else if (lastBoxRef.current) {
      toDraw = lastBoxRef.current;
    }

    if (!toDraw) return;

    const [x, y, w, h] = toDraw.box;

    // letterboxing transform
    const { sx, sy, ox, oy } = computeLetterboxTransform();

    const rx = x * sx + ox;
    const ry = y * sy + oy;
    const rw = w * sx;
    const rh = h * sy;

    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255, 60, 60, 0.95)";
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur = 8;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();
  }

  // Keep canvas sized and redraw while playing
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleResize = () => { resizeCanvasToVideo(); drawOverlay(); };
    const onLoadedMetadata = () => { resizeCanvasToVideo(); drawOverlay(); };
    const onTimeUpdate = () => { drawOverlay(); };
    const onPlay = () => {
      let raf;
      const step = () => {
        drawOverlay();
        if (!video.paused && !video.ended) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
      video._raf = raf;
    };
    const onPause = () => {
      if (video._raf) cancelAnimationFrame(video._raf);
      drawOverlay();
    };

    window.addEventListener("resize", handleResize);
    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);

    // Initial sizing
    handleResize();

    return () => {
      window.removeEventListener("resize", handleResize);
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      if (video._raf) cancelAnimationFrame(video._raf);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boxesMap, fps, metaSize.frame_w, metaSize.frame_h, startFrame, endFrame]);

  return (
    <div style={{ position: "relative", width: "100%", ...style }}>
      <video
        ref={videoRef}
        src={src}
        controls
        style={{ width: "100%", display: "block", borderRadius: 8 }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
