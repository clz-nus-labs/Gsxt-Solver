chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "GSXT_CAPTURE_VISIBLE_TAB") return false;

  const windowId = sender.tab && typeof sender.tab.windowId === "number"
    ? sender.tab.windowId
    : undefined;

  try {
    chrome.tabs.captureVisibleTab(windowId, { format: "png" }, (dataUrl) => {
      if (chrome.runtime.lastError) {
        sendResponse({
          ok: false,
          error: chrome.runtime.lastError.message,
          stage: "captureVisibleTab",
          windowId,
          senderTabId: sender.tab?.id || null,
          senderUrl: sender.url || sender.tab?.url || ""
        });
        return;
      }
      if (!dataUrl) {
        sendResponse({
          ok: false,
          error: "captureVisibleTab returned empty dataUrl",
          stage: "captureVisibleTab",
          windowId
        });
        return;
      }
      sendResponse({ ok: true, dataUrl, windowId });
    });
  } catch (err) {
    sendResponse({
      ok: false,
      error: err?.message || String(err),
      stage: "captureVisibleTab.exception",
      windowId
    });
  }

  return true;
});
