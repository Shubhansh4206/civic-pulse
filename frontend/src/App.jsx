import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, useMap } from "react-leaflet";
import L from "leaflet";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Copy,
  Layers,
  Loader2,
  MapPin,
  Plus,
  RefreshCw,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import "leaflet/dist/leaflet.css";

/* -------------------------------------------------------------------------- */
/*  Config                                                                    */
/* -------------------------------------------------------------------------- */

const API_BASE = "http://127.0.0.1:8000/api/v1";
const MAP_CENTER = [19.076, 72.8777]; // Mumbai
const POLL_MS = 30_000;

const PRIORITY = {
  CRITICAL: { hex: "#ef4444", badge: "bg-red-500 text-white", rank: 3 },
  HIGH: { hex: "#facc15", badge: "bg-yellow-400 text-yellow-950", rank: 2 },
  MEDIUM: { hex: "#fb923c", badge: "bg-orange-400 text-white", rank: 1 },
  LOW: { hex: "#22c55e", badge: "bg-green-500 text-white", rank: 0 },
};
const RESOLVED_HEX = "#9ca3af";
const priorityOf = (level) => PRIORITY[level] ?? PRIORITY.LOW;

const CATEGORY = {
  pothole: { label: "Pothole", Icon: AlertTriangle },
  garbage: { label: "Garbage", Icon: Trash2 },
  garbage_pile: { label: "Garbage Pile", Icon: Trash2 },
};
const categoryOf = (c) =>
  CATEGORY[c?.toLowerCase()] ?? { label: c ?? "Unknown", Icon: AlertTriangle };

const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

/* -------------------------------------------------------------------------- */
/*  Map marker icons (cached so Leaflet doesn't rebuild DOM on every render)  */
/* -------------------------------------------------------------------------- */

const iconCache = new Map();
function getMarkerIcon(color, active) {
  const key = `${color}-${active}`;
  if (iconCache.has(key)) return iconCache.get(key);

  const size = active ? 38 : 26;
  const ring = active
    ? `box-shadow:0 0 0 6px ${color}40, 0 6px 14px rgba(0,0,0,.28);`
    : `box-shadow:0 3px 8px rgba(0,0,0,.25);`;

  const icon = L.divIcon({
    className: "", // drop Leaflet's default white box
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    html: `<div style="
      width:${size}px;height:${size}px;border-radius:9999px;
      background:${color};border:3px solid #fff;${ring}
      transition:all .2s ease;"></div>`,
  });
  iconCache.set(key, icon);
  return icon;
}

/* -------------------------------------------------------------------------- */
/*  Small presentational pieces                                               */
/* -------------------------------------------------------------------------- */

function PriorityBadge({ level, muted = false }) {
  const p = priorityOf(level);
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-[11px] font-bold tracking-wide ${
        muted ? "bg-gray-200 text-gray-500" : p.badge
      }`}
    >
      {level}
    </span>
  );
}

function StatusPill({ status }) {
  const resolved = status === "RESOLVED";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold ${
        resolved ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-600"
      }`}
    >
      {resolved && <CheckCircle2 size={12} />}
      {status === "OPEN" ? "Open" : status === "RESOLVED" ? "Resolved" : status}
    </span>
  );
}

function ScoreBar({ label, value, color }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span className="font-semibold text-slate-800 tabular-nums">{pct(value)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: pct(value), background: color }}
        />
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Ticket card                                                               */
/* -------------------------------------------------------------------------- */

function TicketCard({ ticket, active, onSelect, cardRef }) {
  const { Icon, label } = categoryOf(ticket.category);
  const resolved = ticket.status === "RESOLVED";
  const color = resolved ? RESOLVED_HEX : priorityOf(ticket.priority_level).hex;
  const [copied, setCopied] = useState(false);

  const coords = `${ticket.latitude.toFixed(5)}, ${ticket.longitude.toFixed(5)}`;
  const copyCoords = async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(coords);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <li ref={cardRef}>
      <div
        className={`overflow-hidden rounded-3xl border bg-white transition-all duration-200 ${
          active
            ? "border-slate-900/10 shadow-[0_10px_30px_-10px_rgba(15,23,42,0.25)]"
            : "border-slate-100 hover:border-slate-200 hover:shadow-md"
        } ${resolved ? "opacity-70" : ""}`}
      >
        {/* Header */}
        <button
          type="button"
          onClick={() => onSelect(active ? null : ticket.ticket_id)}
          aria-expanded={active}
          className="flex w-full items-start gap-3 p-4 text-left focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-slate-900"
        >
          <span
            className="mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl"
            style={{ background: `${color}22`, color: resolved ? "#6b7280" : color }}
          >
            <Icon size={20} strokeWidth={2.25} />
          </span>

          <span className="min-w-0 flex-1">
            <span className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-slate-400">
                #{ticket.ticket_id} · {label}
              </span>
              <PriorityBadge level={ticket.priority_level} muted={resolved} />
            </span>
            <span className="mt-1 block truncate text-[15px] font-semibold text-slate-900">
              {ticket.title}
            </span>
            <span className="mt-1 flex items-center gap-2 text-xs text-slate-500">
              <StatusPill status={ticket.status} />
              {ticket.report_count > 1 && (
                <span className="inline-flex items-center gap-1">
                  <Layers size={12} /> {ticket.report_count} reports
                </span>
              )}
            </span>
          </span>

          <ChevronDown
            size={18}
            className={`mt-1 shrink-0 text-slate-400 transition-transform duration-200 ${
              active ? "rotate-180" : ""
            }`}
          />
        </button>

        {/* Expandable details */}
        <div
          className={`grid transition-[grid-template-rows] duration-300 ease-out ${
            active ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
          }`}
        >
          <div className="overflow-hidden">
            <div className="space-y-4 px-4 pb-4">
              {ticket.image_url && (
                <img
                  src={ticket.image_url.startsWith("http") ? ticket.image_url : `http://127.0.0.1:8000/${ticket.image_url}`}
                  alt={ticket.title}
                  loading="lazy"
                  className="h-36 w-full rounded-2xl bg-slate-100 object-cover"
                  onError={(e) => (e.currentTarget.style.display = "none")}
                />
              )}

              <p className="text-sm leading-relaxed text-slate-600">{ticket.description}</p>

              {ticket.duplicate_group_id != null && (
                <div className="flex items-center gap-2 rounded-2xl bg-amber-50 px-3.5 py-2.5 text-xs text-amber-800">
                  <Layers size={14} />
                  <span className="font-semibold">Possible Duplicate</span>
                  <span>{ticket.duplicate_similarity != null ? `${pct(ticket.duplicate_similarity)} similarity · ` : ""}{ticket.related_report_count ?? 0} related reports</span>
                </div>
              )}

              {ticket.predicted_department && (
                <div className="rounded-2xl bg-slate-50 px-3.5 py-2.5 text-xs text-slate-600">
                  <span className="font-semibold text-slate-800">Department: </span>
                  {ticket.predicted_department.replaceAll("_", " ")}
                  {ticket.department_confidence != null && (
                    <span className="ml-2 text-slate-500">
                      ({pct(ticket.department_confidence)} {ticket.routing_method?.endsWith(":low_confidence") ? "best candidate confidence" : "classifier confidence"})
                    </span>
                  )}
                </div>
              )}

              {ticket.clip_status && (
                <div
                  className={`rounded-2xl px-3.5 py-2.5 text-xs ${
                    ticket.clip_status === "CONSISTENT"
                      ? "bg-emerald-50 text-emerald-800"
                      : ticket.clip_status === "SUSPICIOUS"
                      ? "bg-rose-50 text-rose-800"
                      : "bg-slate-50 text-slate-600"
                  }`}
                >
                  <span className="font-semibold text-slate-800">Image-Text Match: </span>
                  <span className="font-medium">{ticket.clip_status}</span>
                  {ticket.clip_similarity != null && (
                    <span className="ml-2 text-slate-500">
                      ({pct(ticket.clip_similarity)} similarity)
                    </span>
                  )}
                </div>
              )}

              <div className="space-y-3 rounded-2xl bg-slate-50 p-3.5">
                <ScoreBar label="Priority score" value={ticket.final_priority_score} color={color} />
                <ScoreBar label="Text urgency" value={ticket.text_urgency} color="#6366f1" />
                <ScoreBar label="Visual severity" value={ticket.visual_severity} color="#0ea5e9" />
              </div>

              <button
                type="button"
                onClick={copyCoords}
                className="flex w-full items-center justify-between rounded-2xl bg-slate-50 px-3.5 py-2.5 text-xs text-slate-500 transition hover:bg-slate-100"
              >
                <span className="inline-flex items-center gap-2">
                  <MapPin size={14} />
                  <span className="font-medium tabular-nums text-slate-700">{coords}</span>
                </span>
                <span className="inline-flex items-center gap-1">
                  <Copy size={12} /> {copied ? "Copied" : "Copy"}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
}

/* -------------------------------------------------------------------------- */
/*  Map helpers                                                               */
/* -------------------------------------------------------------------------- */

function FlyToActive({ ticket }) {
  const map = useMap();
  useEffect(() => {
    if (!ticket) return;
    map.flyTo([ticket.latitude, ticket.longitude], Math.max(map.getZoom(), 15), {
      duration: 0.8,
    });
  }, [ticket?.ticket_id]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

/* -------------------------------------------------------------------------- */
/*  Citizen Report Modal                                                      */
/* -------------------------------------------------------------------------- */

function ReportModal({ isOpen, onClose, onSuccess }) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const removeImage = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImageFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title.trim()) {
      setError("Please enter a title for the hazard.");
      return;
    }
    if (!description.trim()) {
      setError("Please describe the municipal defect.");
      return;
    }
    if (!imageFile) {
      setError("Please select a photo attachment of the hazard.");
      return;
    }

    setSubmitting(true);
    setError(null);

    const formData = new FormData();
    formData.append("title", title.trim());
    formData.append("description", description.trim());
    formData.append("latitude", "19.076");
    formData.append("longitude", "72.8777");
    formData.append("image", imageFile);

    try {
      await axios.post(`${API_BASE}/report`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Clear form
      setTitle("");
      setDescription("");
      removeImage();
      onSuccess();
      onClose();
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        "Failed to submit hazard report. Make sure the backend server is running.";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm">
      <div className="relative w-full max-w-lg rounded-3xl border border-slate-800 bg-slate-900 p-6 text-slate-100 shadow-2xl">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white">Report Municipal Hazard</h2>
            <p className="text-xs text-slate-400">
              Submit photo and details for automated YOLOv8 CV & NLP priority triage.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="flex h-9 w-9 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-800 hover:text-white"
          >
            <X size={18} />
          </button>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-2xl bg-red-950/80 p-3 text-xs text-red-200 border border-red-800">
            <AlertTriangle size={15} className="mt-0.5 shrink-0 text-red-400" />
            <span className="flex-1">{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Title */}
          <div>
            <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Hazard Title
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Deep pothole on 5th Avenue"
              disabled={submitting}
              className="w-full rounded-2xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-white placeholder-slate-500 transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          {/* Description */}
          <div>
            <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Provide context (e.g. Large open hole obstructing lane, high risk of accident)..."
              rows={3}
              disabled={submitting}
              className="w-full resize-none rounded-2xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-white placeholder-slate-500 transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          {/* Image Picker */}
          <div>
            <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Photo Evidence (.jpg, .png)
            </label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png, image/jpeg, image/jpg, image/webp"
              onChange={handleImageChange}
              className="hidden"
            />

            {!previewUrl ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-800/40 p-5 transition hover:border-slate-500 hover:bg-slate-800/70"
              >
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-slate-800 text-slate-300">
                  <UploadCloud size={22} />
                </div>
                <p className="mt-2 text-xs font-medium text-slate-300">
                  Click to upload or drag & drop hazard photo
                </p>
                <p className="mt-0.5 text-[11px] text-slate-500">JPG, PNG, or WEBP up to 10MB</p>
              </div>
            ) : (
              <div className="relative overflow-hidden rounded-2xl border border-slate-700 bg-slate-950">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="h-40 w-full object-cover"
                />
                <button
                  type="button"
                  onClick={removeImage}
                  className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full bg-slate-900/80 text-white backdrop-blur-sm transition hover:bg-red-600"
                >
                  <X size={15} />
                </button>
                <div className="absolute bottom-2 left-2 rounded-lg bg-slate-900/90 px-2.5 py-1 text-[11px] font-medium text-slate-300 backdrop-blur-sm">
                  {imageFile?.name}
                </div>
              </div>
            )}
          </div>

          {/* Coordinates Tag */}
          <div className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-800/40 px-3.5 py-2.5 text-xs text-slate-400">
            <span className="flex items-center gap-2">
              <MapPin size={14} className="text-indigo-400" />
              <span>Location: 19.07600, 72.87770</span>
            </span>
            <span className="rounded-md bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-400">
              Auto-Tagged (Mumbai)
            </span>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-3">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="rounded-full px-5 py-2.5 text-sm font-semibold text-slate-400 transition hover:bg-slate-800 hover:text-white"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-600/30 transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {submitting ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Submitting & Analyzing...</span>
                </>
              ) : (
                <>
                  <CheckCircle2 size={16} />
                  <span>Submit Hazard</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Main App Component                                                        */
/* -------------------------------------------------------------------------- */

export default function App() {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [resolving, setResolving] = useState(false);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const cardRefs = useRef(new Map());

  /* ---- data fetching ---- */
  const fetchTickets = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const { data } = await axios.get(`${API_BASE}/report`);
      setTickets(Array.isArray(data) ? data : []);
      setError(null);
    } catch (err) {
      if (!silent) {
        setError(
          err.response
            ? `The server responded with an error (${err.response.status}).`
            : "Can't reach the CivicPulse API. Check that the backend is running on port 8000."
        );
      }
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTickets();
    const id = setInterval(() => fetchTickets(true), POLL_MS);
    return () => clearInterval(id);
  }, [fetchTickets]);

  /* ---- derived ---- */
  const sorted = useMemo(
    () => [...tickets].sort((a, b) => b.final_priority_score - a.final_priority_score),
    [tickets]
  );
  const activeTicket = useMemo(
    () => tickets.find((t) => t.ticket_id === activeId) ?? null,
    [tickets, activeId]
  );
  const openCount = useMemo(() => tickets.filter((t) => t.status !== "RESOLVED").length, [tickets]);

  /* ---- scroll sidebar card into view ---- */
  useEffect(() => {
    if (activeId == null) return;
    const el = cardRefs.current.get(activeId);
    const t = setTimeout(() => el?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 60);
    return () => clearTimeout(t);
  }, [activeId]);

  /* ---- resolve action ---- */
  const resolveTicket = async (ticket) => {
    setResolving(true);
    const prev = tickets;
    setTickets((ts) =>
      ts.map((t) => (t.ticket_id === ticket.ticket_id ? { ...t, status: "RESOLVED" } : t))
    );
    try {
      await axios.patch(`${API_BASE}/report/${ticket.ticket_id}`, { status: "RESOLVED" });
    } catch {
      setTickets(prev);
      setError("Couldn't mark the ticket as resolved. Try again.");
    } finally {
      setResolving(false);
    }
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-slate-200 font-sans text-slate-900">
      {/* Full-screen Free OpenStreetMap */}
      <MapContainer
        center={MAP_CENTER}
        zoom={12}
        zoomControl={false}
        className="absolute inset-0 z-0 h-full w-full"
        style={{ background: "#e5e7eb" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
        />

        {sorted.map((t) => {
          const color = t.status === "RESOLVED" ? RESOLVED_HEX : priorityOf(t.priority_level).hex;
          const isActive = t.ticket_id === activeId;
          return (
            <Marker
              key={t.ticket_id}
              position={[t.latitude, t.longitude]}
              icon={getMarkerIcon(color, isActive)}
              zIndexOffset={isActive ? 1000 : Math.round(t.final_priority_score * 100)}
              eventHandlers={{ click: () => setActiveId(t.ticket_id) }}
            />
          );
        })}

        <FlyToActive ticket={activeTicket} />
      </MapContainer>

      {/* Floating Dispatcher Sidebar */}
      <aside className="pointer-events-none absolute inset-x-3 top-3 z-[1000] flex max-h-[46vh] md:inset-x-auto md:bottom-4 md:left-4 md:top-4 md:max-h-none md:w-[24rem]">
        <div className="pointer-events-auto flex w-full flex-col overflow-hidden rounded-[2rem] bg-white shadow-[0_20px_60px_-15px_rgba(15,23,42,0.35)]">
          <header className="flex items-center justify-between px-6 pb-3 pt-6">
            <div>
              <h1 className="text-xl font-bold tracking-tight">CivicPulse</h1>
              <p className="text-sm text-slate-500">
                {loading ? "Loading reports…" : `${openCount} open of ${tickets.length} reports`}
              </p>
            </div>
            <button
              type="button"
              onClick={() => fetchTickets()}
              aria-label="Refresh tickets"
              className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-600 transition hover:bg-slate-200 focus-visible:outline-2 focus-visible:outline-slate-900"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
            </button>
          </header>

          {error && (
            <div
              role="alert"
              className="mx-4 mb-2 flex items-start gap-2 rounded-2xl bg-red-50 px-3.5 py-3 text-sm text-red-700"
            >
              <AlertTriangle size={16} className="mt-0.5 shrink-0" />
              <span className="flex-1">{error}</span>
              <button onClick={() => setError(null)} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}

          <ul className="flex-1 space-y-3 overflow-y-auto px-4 pb-4 pt-1 [scrollbar-width:thin]">
            {loading && tickets.length === 0 &&
              [0, 1, 2].map((i) => (
                <li key={i} className="h-24 animate-pulse rounded-3xl bg-slate-100" />
              ))}

            {!loading && !error && tickets.length === 0 && (
              <li className="px-4 py-10 text-center text-sm text-slate-500">
                No hazards reported yet. New reports appear here as they come in.
              </li>
            )}

            {sorted.map((t) => (
              <TicketCard
                key={t.ticket_id}
                ticket={t}
                active={t.ticket_id === activeId}
                onSelect={setActiveId}
                cardRef={(el) => {
                  if (el) cardRefs.current.set(t.ticket_id, el);
                  else cardRefs.current.delete(t.ticket_id);
                }}
              />
            ))}
          </ul>
        </div>
      </aside>

      {/* Floating Bottom Ticket Summary Panel */}
      {activeTicket && (
        <section
          aria-label="Selected ticket"
          className="absolute inset-x-3 bottom-24 z-[1000] md:bottom-6 md:left-[26.5rem] md:right-56 md:max-w-xl"
        >
          <div className="flex items-center gap-4 rounded-[2rem] bg-white p-3 pr-4 shadow-[0_20px_60px_-15px_rgba(15,23,42,0.4)]">
            {activeTicket.image_url ? (
              <img
                src={activeTicket.image_url.startsWith("http") ? activeTicket.image_url : `http://127.0.0.1:8000/${activeTicket.image_url}`}
                alt=""
                className="hidden h-16 w-16 shrink-0 rounded-3xl bg-slate-100 object-cover sm:block"
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
            ) : null}

            <div className="min-w-0 flex-1 pl-2 sm:pl-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-bold">#{activeTicket.ticket_id}</span>
                <PriorityBadge
                  level={activeTicket.priority_level}
                  muted={activeTicket.status === "RESOLVED"}
                />
                <StatusPill status={activeTicket.status} />
              </div>
              <p className="mt-1 truncate text-sm font-medium text-slate-800">
                {activeTicket.title}
              </p>
              <p className="text-xs text-slate-500">
                {categoryOf(activeTicket.category).label} · Priority{" "}
                {pct(activeTicket.final_priority_score)}
              </p>
            </div>

            <button
              type="button"
              disabled={activeTicket.status === "RESOLVED" || resolving}
              onClick={() => resolveTicket(activeTicket)}
              className="inline-flex shrink-0 items-center gap-2 rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
            >
              <CheckCircle2 size={16} />
              {activeTicket.status === "RESOLVED" ? "Resolved" : resolving ? "Resolving…" : "Resolve"}
            </button>

            <button
              type="button"
              onClick={() => setActiveId(null)}
              aria-label="Close summary"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
            >
              <X size={18} />
            </button>
          </div>
        </section>
      )}

      {/* Floating Action Button (FAB): Report Hazard */}
      <button
        type="button"
        onClick={() => setIsReportModalOpen(true)}
        className="fixed bottom-6 right-6 z-[1050] flex items-center gap-3 rounded-full bg-slate-900 px-5 py-3.5 text-sm font-semibold text-white shadow-[0_12px_32px_rgba(15,23,42,0.45)] transition-all duration-200 hover:scale-105 hover:bg-slate-800 active:scale-95 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-indigo-600 text-white shadow-sm">
          <Plus size={16} strokeWidth={3} />
        </span>
        <span className="tracking-wide">Report Hazard</span>
      </button>

      {/* Citizen Reporting Modal */}
      <ReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        onSuccess={fetchTickets}
      />
    </div>
  );
}
