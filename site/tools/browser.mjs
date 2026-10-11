// A small driver for headless Chrome over its DevTools protocol, for the
// tools here that need a browser to draw a page: no dependency, only the
// Chrome that is installed. CHROME names another binary.

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

export async function browser() {
  // Chrome picks a free port and writes it down, so that several can run at once.
  const profile = mkdtempSync(join(tmpdir(), "minllm-"));
  const child = spawn(CHROME, ["--headless=new", "--hide-scrollbars", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
  const note = join(profile, "DevToolsActivePort");
  for (let tries = 0; !existsSync(note) || !readFileSync(note, "utf8").includes("\n"); tries++) {
    if (tries > 100) throw new Error(`no answer from ${CHROME}`);
    await new Promise((r) => setTimeout(r, 100));
  }
  const [port, path] = readFileSync(note, "utf8").split("\n");
  const socket = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  await new Promise((resolve) => (socket.onopen = resolve));
  let sent = 0;
  const waiting = new Map();
  const thrown = []; // what went wrong in a page: [session, text]
  socket.onmessage = (message) => {
    const { id, result, error, method, params, sessionId } = JSON.parse(message.data);
    if (method === "Runtime.exceptionThrown") thrown.push([sessionId, params.exceptionDetails.exception?.description || params.exceptionDetails.text]);
    if (!waiting.has(id)) return;
    const { resolve, reject } = waiting.get(id);
    waiting.delete(id);
    error ? reject(new Error(error.message)) : resolve(result);
  };
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    waiting.set(++sent, { resolve, reject });
    socket.send(JSON.stringify({ id: sent, method, params, sessionId }));
  });

  // A page of a given size. scale is how many pixels of a picture one CSS pixel becomes.
  async function page({ width, height, scale = 1 }) {
    const { targetId } = await send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
    const call = (method, params) => send(method, params, sessionId);
    await call("Page.enable");
    await call("Runtime.enable");
    await call("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: scale, mobile: false });
    return {
      async open(url) {
        await call("Page.navigate", { url });
      },
      // The value of an expression in the page. A promise is waited for.
      async run(expression) {
        const { result, exceptionDetails } = await call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
        if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || exceptionDetails.text);
        return result.value;
      },
      // What was thrown in the page and caught by nobody.
      errors: () => thrown.filter(([session]) => session === sessionId).map(([, text]) => text),
      // The page as it is drawn now, as the bytes of a PNG.
      async picture() {
        return Buffer.from((await call("Page.captureScreenshot", { format: "png" })).data, "base64");
      },
    };
  }
  return { page, close: () => (socket.close(), child.kill()) };
}
