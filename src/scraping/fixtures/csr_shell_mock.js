// Simulates a client-side-rendered voucher page.
// When Playwright loads csr_shell_mock.html and executes this script,
// it injects a voucher card into #app — proving JS rendering is active.
// A raw HTTP GET of the HTML file would see only <div id="app"></div>.
(function () {
  var app = document.getElementById("app");
  if (!app) return;

  var card = document.createElement("div");
  card.className = "voucher-card";

  var merchant = document.createElement("span");
  merchant.className = "merchant-name";
  merchant.textContent = "StreamGear";

  var title = document.createElement("p");
  title.className = "voucher-title";
  title.textContent = "Exclusive affiliate promo — 99p off";

  var code = document.createElement("code");
  code.className = "voucher-code";
  code.textContent = "CSRLEAK99";

  card.appendChild(merchant);
  card.appendChild(title);
  card.appendChild(code);
  app.appendChild(card);
})();
