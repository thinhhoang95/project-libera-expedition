import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { api } from "./api";
import { Layout } from "./components/Layout";
import { ReviewerProvider } from "./context";
import "./index.css";
import { Experiments } from "./pages/Experiments";
import { Explore } from "./pages/Explore";
import { MapGrid } from "./pages/MapGrid";
import { MyPaper } from "./pages/MyPaper";
import { NewProject } from "./pages/NewProject";
import { Pipeline, Queue } from "./pages/Pipeline";
import { Projects } from "./pages/Projects";
import { ResearchLab } from "./pages/ResearchLab";
import { ClusterReview, Corpus, FacetReview } from "./pages/Review";
import { StartHere } from "./pages/StartHere";

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
