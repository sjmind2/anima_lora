(function () {
  var Vue = window.Vue;
  var SUPPORTED = ["zh-CN", "en", "ja"];
  var STORAGE_KEY = "anima-locale";
  var _messages = Vue.reactive({});
  var _fallback = {};
  var _locale = "en";

  function _detectLocale() {
    var stored = null;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    if (stored && SUPPORTED.indexOf(stored) >= 0) return stored;
    var nav = (navigator.language || navigator.userLanguage || "en").toLowerCase();
    if (nav.indexOf("zh") === 0) return "zh-CN";
    if (nav.indexOf("ja") === 0) return "ja";
    return "en";
  }

  function _resolve(obj, parts) {
    for (var i = 0; i < parts.length; i++) {
      if (obj == null || typeof obj !== "object") return undefined;
      obj = obj[parts[i]];
    }
    return obj;
  }

  function _interpolate(template, params) {
    if (!params) return template;
    return template.replace(/\{(\w+)\}/g, function (m, k) {
      return params[k] !== undefined ? String(params[k]) : m;
    });
  }

  function _lookup(key) {
    var parts = key.split(".");
    var val = _resolve(_messages, parts);
    if (val !== undefined && typeof val === "string") return val;
    val = _resolve(_fallback, parts);
    if (val !== undefined && typeof val === "string") return val;
    return null;
  }

  function _setMessages(data) {
    Object.keys(_messages).forEach(function(k) { delete _messages[k]; });
    Object.assign(_messages, data);
  }

  window.t = function (key, params) {
    var val = _lookup(key);
    if (val === null) return key;
    return _interpolate(val, params);
  };

  window.I18n = {
    init: function () {
      _locale = _detectLocale();
      var fallbackUrl = "/static/i18n/locales/en.json";
      var localeUrl = "/static/i18n/locales/" + _locale + ".json";
      return fetch(fallbackUrl)
        .then(function (r) { return r.json(); })
        .then(function (data) { _fallback = data; })
        .then(function () {
          if (_locale === "en") { _setMessages(_fallback); return; }
          return fetch(localeUrl)
            .then(function (r) { return r.json(); })
            .then(function (data) { _setMessages(data); })
            .catch(function () { _setMessages(_fallback); });
        });
    },
    setLocale: function (locale) {
      if (SUPPORTED.indexOf(locale) < 0) return Promise.resolve();
      _locale = locale;
      try { localStorage.setItem(STORAGE_KEY, locale); } catch (e) {}
      if (locale === "en") {
        _setMessages(_fallback);
        return Promise.resolve();
      }
      return fetch("/static/i18n/locales/" + locale + ".json")
        .then(function (r) { return r.json(); })
        .then(function (data) { _setMessages(data); })
        .catch(function () { _setMessages(_fallback); });
    },
    getLocale: function () { return _locale; }
  };
})();
