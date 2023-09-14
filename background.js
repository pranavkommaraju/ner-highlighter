chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message.action === 'togglePer') {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      const activeTab = tabs[0];
      if (activeTab) {
        chrome.tabs.sendMessage(activeTab.id, { action: 'togglePer' });
      }
    });
  }
  if (message.action === 'toggleLoc') {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        const activeTab = tabs[0];
        if (activeTab) {
            chrome.tabs.sendMessage(activeTab.id, { action: 'toggleLoc' });
        }
    });
  }
  if (message.action === 'toggleOrg') {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        const activeTab = tabs[0];
        if (activeTab) {
            chrome.tabs.sendMessage(activeTab.id, { action: 'toggleOrg' });
        }
    });
  }
  if (message.action === 'toggleMisc') {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        const activeTab = tabs[0];
        if (activeTab) {
            chrome.tabs.sendMessage(activeTab.id, { action: 'toggleMisc' });
        }
    });
  }
});

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  fetch("http://localhost:8000/ner_extraction_app/extract_entities/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: request.text }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Entities received from server:", data.entities);
      chrome.tabs.sendMessage(sender.tab.id, { entities: data.entities });
    });
});
