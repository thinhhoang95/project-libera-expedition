import clsx from "clsx";
import { Children, memo, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useCategorySheet, usePaperDrawer } from "../context";

export const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const textOf = (n: ReactNode): string => Children.toArray(n).map((c: any) => (typeof c === "string" ? c : textOf(c?.props?.children))).join("");
const PLUGINS = [remarkGfm];

/** Markdown from the report's model: [title](paper:W…) opens the paper card, [label](cat:<cluster id>) the details sheet. */
export const ReportMarkdown = memo(function ReportMarkdown({ md, className }: { md: string; className?: string }) {
  const drawer = usePaperDrawer();
  const sheet = useCategorySheet();
  return (
    <div className={clsx("prose-sm", className)}>
      <ReactMarkdown remarkPlugins={PLUGINS}
        urlTransform={(u) => (/^(paper|cat):/.test(u) || /^(https?:|mailto:|#)/.test(u) ? u : "")}
        components={{
          h2: ({ children }) => <h2 id={slug(textOf(children))} className="scroll-mt-24">{children}</h2>,
          a: ({ href, children }) => {
            if (href?.startsWith("paper:")) return <button className="ref-link" onClick={() => drawer.open(href.slice(6))}>{children}</button>;
            if (href?.startsWith("cat:")) return <button className="ref-link cat" onClick={() => sheet.open(href.slice(4))}>{children}</button>;
            return href ? <a href={href} target="_blank" rel="noreferrer">{children}</a> : <span>{children}</span>;
          },
        }}>{md}</ReactMarkdown>
    </div>
  );
});
