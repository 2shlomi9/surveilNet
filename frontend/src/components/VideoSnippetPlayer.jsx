import { useEffect, useRef, useState } from "react";

/**
 * VideoSnippetPlayer
 *
 * Draws a moving face box overlay (one box per frame, best score above threshold)
 * on top of a snippet <video>. Boxes are fetched from /api/match_boxes.
 *
 * Props:
 *  - src: string                      // URL to the snippet video (mp4)
 *  - personId: string                 // person id to match against
 *  - videoName: string                // original video filename (used by frame_store/<name_noext>)
 *  - startFrame: number               // first frame index of the snippet in the original video
 *  - endFrame: number                 // last frame index (inclusive)
 *  - fps: number                      // fps used to map currentTime -> frame index
 *  - threshold?: number               // optional override for server-side threshold
 *  - style?: React.CSSProperties      // optional wrapper styles
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

  const [boxesMap, setBoxesMap] = useState(new Map()); // frame_idx -> { box:[x,y,w,h], score:number }
  const [metaSize, setMetaSize] = useState({ frame_w: null, frame_h: null, fps_meta: null });

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
        if (typeof threshold === "number") {
          params.set("threshold", String(threshold));
        }

        const res = await fetch(`http://localhost:5000/api/match_boxes?${params.toString()}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load boxes");

        if (abort) return;

        const map = new Map();
        for (const r of data.boxes || []) {
          map.set(Number(r.frame_idx), { box: r.box, score: r.score });
        }
        setBoxesMap(map);
        setMetaSize({
          frame_w: data.frame_h && data.frame_w ? Number(data.frame_w) : null,
          frame_h: data.frame_h && data.frame_w ? Number(data.frame_h) : null,
          fps_meta: typeof data.fps === "number" ? data.fps : null,
        });
      } catch (e) {
        console.warn("[VideoSnippetPlayer] overlay fetch failed:", e);
        setBoxesMap(new Map());
        setMetaSize({ frame_w: null, frame_h: null, fps_meta: null });
      }
    }

    fetchBoxes();
    return () => { abort = true; };
  }, [personId, videoName, startFrame, endFrame, threshold]);

  // Resize canvas to match the displayed size of the video element
  function resizeCanvasToVideo() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const rect = video.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
  }

  // Draw one frame's overlay
  function drawOverlay() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear previous overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Compute current frame index in the original video
    const currentTime = video.currentTime || 0;
    const curOffsetFrames = Math.round(currentTime * fps);
    const curFrameIdx = startFrame + curOffsetFrames;

    const entry = boxesMap.get(curFrameIdx);
    if (!entry) return;

    const [x, y, w, h] = entry.box.map(Number);

    // Scale from original frame size to displayed size
    const frameW = metaSize.frame_w || video.videoWidth || canvas.width;
    const frameH = metaSize.frame_h || video.videoHeight || canvas.height;
    const sx = canvas.width / frameW;
    const sy = canvas.height / frameH;

    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255, 60, 60, 0.95)";
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur = 8;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    ctx.restore();
  }

  // Keep canvas sized and redraw while playing
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleResize = () => {
      resizeCanvasToVideo();
      drawOverlay();
    };

    const onLoadedMetadata = () => {
      resizeCanvasToVideo();
      drawOverlay();
    };

    const onTimeUpdate = () => {
      // Fallback draw (fires ~4-5x/sec). We also do RAF while playing for smoothness.
      drawOverlay();
    };

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

    // Initial sizing in case metadata is already available
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
