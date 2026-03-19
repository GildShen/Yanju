import { useEffect } from "react";

function App() {
  useEffect(() => {
    window.location.replace("/legacy.html");
  }, []);

  return null;
}

export default App;
