import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, lazy, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { api } from "./api";
import { Layout } from "./components/Layout";
import { ReviewerProvider } from "./context";
import "./index.css";

// pages load when first opened, so the first screen does not wait for every view (and Plotly) to parse
const page = <K extends string>(load: () => Promise<Record<K, React.ComponentType>>, name: K) =>
  lazy(() => load().then((m) => ({ default: m[name] })));
const Experiments = page(() => import("./pages/Experiments"), "Experiments");
const FieldReport = page(() => import("./pages/FieldReport"), "FieldReport");
const Explore = page(() => import("./pages/Explore"), "Explore");
const MapGrid = page(() => import("./pages/MapGrid"), "MapGrid");
const MyPaper = page(() => import("./pages/MyPaper"), "MyPaper");
const NewProject = page(() => import("./pages/NewProject"), "NewProject");
const Pipeline = page(() => import("./pages/Pipeline"), "Pipeline");
const Queue = page(() => import("./pages/Pipeline"), "Queue");
const Projects = page(() => import("./pages/Projects"), "Projects");
const ResearchLab = page(() => import("./pages/ResearchLab"), "ResearchLab");
const ClusterReview = page(() => import("./pages/Review"), "ClusterReview");
const Corpus = page(() => import("./pages/Review"), "Corpus");
const FacetReview = page(() => import("./pages/Review"), "FacetReview");
const StartHere = page(() => import("./pages/StartHere"), "StartHere");

const qc = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } });

/** Remembers the last opened project (sorts it first on Projects). */
function ProjectScope() {
  const { project } = useParams();
  useEffect(() => {
    if (project) api(`/projects/${project}`, { method: "PATCH", json: { last_opened: new Date().toISOString() } }).catch(() => {});
  }, [project]);
  return null;
}

function P({ children }: { children: React.ReactNode }) {
  return <><ProjectScope />{children}</>;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <ReviewerProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Projects />} />
                <Route path="new" element={<NewProject />} />
                <Route path="queue" element={<Queue />} />
                <Route path="p/:project">
                  <Route index element={<Navigate to="start" replace />} />
                  <Route path="start" element={<P><StartHere /></P>} />
                  <Route path="paper" element={<P><MyPaper /></P>} />
                  <Route path="paper/:pid" element={<P><MyPaper /></P>} />
                  <Route path="report" element={<P><FieldReport /></P>} />
                  <Route path="explore" element={<P><Explore /></P>} />
                  <Route path="explore/:facet" element={<P><Explore /></P>} />
                  <Route path="map" element={<P><MapGrid /></P>} />
                  <Route path="lab" element={<P><ResearchLab /></P>} />
                  <Route path="experiments" element={<P><Experiments /></P>} />
                  <Route path="corpus" element={<P><Corpus /></P>} />
                  <Route path="facets" element={<P><FacetReview /></P>} />
                  <Route path="clusters" element={<P><ClusterReview /></P>} />
                  <Route path="pipeline" element={<P><Pipeline /></P>} />
                </Route>
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
      </ReviewerProvider>
    </QueryClientProvider>
  </StrictMode>,
);
