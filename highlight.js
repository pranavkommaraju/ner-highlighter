const entityStyles = {
    PER: 'highlighted-per',
    LOC: 'highlighted-loc',
    ORG: 'highlighted-org',
    MISC: 'highlighted-misc',
};

function highlightEntities(entities) {
    entities.forEach(entity => {
        const { type, startIndex, endIndex } = entity;
        const selectedText = window.getSelection().toString();
        console.log(entity.text)
        if (selectedText === entity.text) {
            console.log("matched entity")
            const span = document.createElement('span');
            span.className = entityStyles[type];
            span.textContent = entity.text;

            const range = window.getSelection().getRangeAt(0);
            range.deleteContents();
            range.insertNode(span);
        }
    });
}

function toggleHighlight(entityType) {
    console.log("toggled")
    const highlightedElements = document.getElementsByClassName(entityStyles[entityType]);
    for (let i = 0; i < highlightedElements.length; i++) {
        const element = highlightedElements[i];
        if (element.style.display === 'none') {
            element.style.display = '';
        } else {
            element.style.display = 'none';
        }
    }
}

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
    if (message.action === 'togglePer') {
        toggleHighlight('PER');
    }
    if (message.action === 'toggleLoc') {
        toggleHighlight('LOC');
    }
    if (message.action === 'toggleOrg') {
        toggleHighlight('ORG');
    }
    if (message.action === 'toggleMisc') {
        toggleHighlight('MISC');
    }
});





document.addEventListener("DOMContentLoaded", function () {
    console.log('started getting entities')
    const text = document.body.textContent;
    chrome.runtime.sendMessage({ text }, function (response) {
        console.log("Entities received:", response.entities);
        highlightEntities(response.entities)
    });
});
  
  