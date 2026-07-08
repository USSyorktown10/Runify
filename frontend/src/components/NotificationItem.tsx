import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { AthleteAvatar } from "@/components/AthleteAvatar";
import { ClubAvatar } from "@/components/ClubAvatar";
import { formatDateTime } from "@/lib/format";
import type { Notification, NotificationTarget } from "@/types/api";

function ActivityTypeIcon({ type, className = "h-5 w-5 text-accent" }: { type: string; className?: string }) {
  const t = type.toLowerCase();
  if (t === "run") {
    return (
      <svg fill="currentColor" viewBox="0 0 24 24" className={className} aria-hidden>
        <path d="M8.688 0C8.025 0 7.38.215 6.85.613l-3.32 2.49-2.845.948A1 1 0 000 5c0 1.579.197 2.772.567 3.734.376.978.907 1.654 1.476 2.223.305.305.6.567.886.82.785.697 1.5 1.33 2.159 2.634 1.032 2.57 2.37 4.748 4.446 6.27C11.629 22.218 14.356 23 18 23c2.128 0 3.587-.553 4.549-1.411a4.378 4.378 0 001.408-2.628c.152-.987-.389-1.787-.967-2.25l-3.892-3.114a1 1 0 01-.329-.477l-3.094-9.726A2 2 0 0013.769 2h-1.436a2 2 0 00-1.2.4l-.57.428-.516-1.803A1.413 1.413 0 008.688 0z" />
      </svg>
    );
  }
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
    </svg>
  );
}

function TargetPreview({ target }: { target: NotificationTarget }) {
  let leadingVisual: ReactNode = null;

  if (target.kind === "club") {
    leadingVisual = (
      <ClubAvatar
        club={{ id: target.id, name: target.title, profile_picture_url: target.image_url ?? "" }}
        size="sm"
      />
    );
  } else if (target.kind === "athlete") {
    const parts = target.title.trim().split(/\s+/);
    leadingVisual = (
      <AthleteAvatar
        athlete={{
          id: target.id,
          first_name: parts[0] ?? "",
          last_name: parts.slice(1).join(" "),
          profile_picture_url: target.image_url ?? "",
        }}
        size="sm"
      />
    );
  } else if (target.kind === "activity" && target.activity_type) {
    leadingVisual = (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center border border-border bg-global-bg">
        <ActivityTypeIcon type={target.activity_type} />
      </div>
    );
  } else if (target.kind === "post") {
    leadingVisual = (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center border border-border bg-global-bg text-lg">
        📝
      </div>
    );
  }

  return (
    <div className="mt-2 flex gap-2.5 border border-border/60 bg-surface/50 p-2.5">
      {leadingVisual}

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold text-global-text">{target.title}</p>
        {target.subtitle && <p className="mt-0.5 truncate text-xs text-muted">{target.subtitle}</p>}
        {target.detail && (
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted">{target.detail}</p>
        )}
      </div>
    </div>
  );
}

export function NotificationItem({
  notification,
  onNavigate,
}: {
  notification: Notification;
  onNavigate?: (notification: Notification) => void;
}) {
  const sender = notification.sender;

  return (
    <Link
      to={notification.link_path}
      onClick={() => onNavigate?.(notification)}
      className={`card flex gap-3 items-start transition-colors hover:bg-surface ${
        notification.is_read ? "text-muted" : "border-accent"
      }`}
    >
      <div className="shrink-0 pt-0.5">
        {sender ? (
          <AthleteAvatar athlete={sender} size="sm" />
        ) : (
          <div className="flex h-8 w-8 items-center justify-center border-2 border-border bg-surface text-sm">
            🔔
          </div>
        )}
      </div>

      <div className="min-w-0 flex-1">
        <p className={`text-sm leading-snug ${notification.is_read ? "" : "font-semibold text-global-text"}`}>
          {notification.message}
        </p>

        {notification.excerpt && (
          <blockquote className="mt-2 border-s-2 border-accent/60 ps-2.5 text-sm italic text-global-text/90">
            &ldquo;{notification.excerpt}&rdquo;
          </blockquote>
        )}

        {notification.target && <TargetPreview target={notification.target} />}

        <time className="mt-2 block text-xs text-muted">{formatDateTime(notification.created_at)}</time>
      </div>

      {!notification.is_read && (
        <span className="mt-2 h-2 w-2 shrink-0 rounded-full bg-accent" aria-label="Unread" />
      )}
    </Link>
  );
}
