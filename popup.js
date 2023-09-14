document.addEventListener('DOMContentLoaded', function () {
  const perButton = document.getElementById('perButton');
  const locButton = document.getElementById('locButton');
  const orgButton = document.getElementById('orgButton');
  const miscButton = document.getElementById('miscButton');

  let show_per = true;
  let show_loc = true;
  let show_org = true;
  let show_misc = true;

  function per_entity() {
    show_per = !show_per;
    perButton.textContent = show_per ? 'Persons Highlighted' : 'Persons Not Highlighted';
    chrome.runtime.sendMessage({ action: 'togglePer' });
  }

  function loc_entity() {
    show_loc = !show_loc;
    locButton.textContent = show_loc ? 'Locations Highlighted' : 'Locations Not Highlighted';
    chrome.runtime.sendMessage({ action: 'toggleLoc' });
  }

  function org_entity() {
    show_org = !show_org;
    orgButton.textContent = show_org ? 'Organizations Highlighted' : 'Organizations Not Highlighted';
    chrome.runtime.sendMessage({ action: 'toggleOrg' });
  }

  function misc_entity() {
    show_misc = !show_misc;
    miscButton.textContent = show_misc ? 'Miscellaneous Highlighted' : 'Miscellaneous Not Highlighted';
    chrome.runtime.sendMessage({ action: 'toggleMisc' });
  }

  perButton.addEventListener('click', per_entity);
  locButton.addEventListener('click', loc_entity);
  orgButton.addEventListener('click', org_entity);
  miscButton.addEventListener('click', misc_entity);
});
  