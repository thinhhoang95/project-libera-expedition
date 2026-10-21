import { useEffect, useRef } from "react";

// open overlays, innermost last: only the top one traps focus; the page scrolls again once the last one closes
const stack: HTMLElement[] = [];

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function lock() {
  const b = document.body;
  if (stack.length === 1) {
    const gap = window.innerWidth - document.documentElement.clientWidth;
    b.style.overflow = "hidden";
    if (gap > 0) b.style.paddingRight = `${gap}px`; // no layout jump when the scrollbar goes
  }
}

function unlock() {
  if (stack.length === 0) {
    document.body.style.overflow = "";
    document.body.style.paddingRight = "";
  }
}

/**
 * A modal surface (sheet, drawer, palette): locks the page scroll behind it, moves focus into it, keeps Tab
 * inside it while it is the top overlay, and gives focus back to what had it when it closes.
 */
export function useOverlay<T extends HTMLElement>(open: boolean, opts: { autoFocus?: boolean } = {}) {
  const ref = useRef<T>(null);
  useEffect(() => {
    const el = ref.current;
    if (!open || !el) return;
    const before = document.activeElement as HTMLElement | null;
    stack.push(el);
    lock();
    if (opts.autoFocus !== false && !el.contains(document.activeElement)) el.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || stack[stack.length - 1] !== el) return;
      const items = [...el.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((x) => x.offsetParent !== null);
      if (!items.length) return;
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && (document.activeElement === first || document.activeElement === el)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      const i = stack.indexOf(el);
      if (i >= 0) stack.splice(i, 1);
      unlock();
      if (before?.isConnected) before.focus({ preventScroll: true });
    };
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  return ref;
}

/** True when `el` is the overlay on top (Escape should close only that one). */
export const isTopOverlay = (el: HTMLElement | null) => !!el && stack[stack.length - 1] === el;
