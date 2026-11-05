import { useEffect, useState } from "react";
import { Home } from "./Home";
import { Review } from "./Review";

export default function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const change = () => setHash(location.hash);
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  const route = /^#(run|metrics)\/([a-f0-9]{32})$/.exec(hash);
  return (
    <>
      <header>
        <a className="logo" href="#">
          Moto<span>ReID</span>
        </a>
        <a href="#">Run workspace</a>
      </header>
      {route ? (
        <Review id={route[2]} metrics={route[1] === "metrics"} />
      ) : (
        <Home />
      )}
    </>
  );
}
