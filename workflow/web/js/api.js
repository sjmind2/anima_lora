var AnimaAPI = (function () {
  var BASE = "";

  function request(method, path, body) {
    var opts = {
      method: method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
    }
    if (typeof I18n !== "undefined") {
      opts.headers["Accept-Language"] = I18n.getLocale();
    }
    return fetch(BASE + path, opts).then(function (r) {
      if (!r.ok) {
        return r.json().then(
          function (err) {
            throw err;
          },
          function () {
            throw { error: r.statusText, status: r.status };
          }
        );
      }
      if (r.status === 204) return null;
      return r.json();
    });
  }

  function get(path) {
    return request("GET", path);
  }

  function post(path, body) {
    return request("POST", path, body);
  }

  function put(path, body) {
    return request("PUT", path, body);
  }

  function del(path) {
    return request("DELETE", path);
  }

  function connectEventStream(runId, onEvent) {
    var url = BASE + "/api/runs/" + encodeURIComponent(runId) + "/events";
    var source = new EventSource(url);
    source.onmessage = function (e) {
      try {
        var data = JSON.parse(e.data);
        onEvent(data);
      } catch (ex) {
        console.error("SSE parse error:", ex);
      }
    };
    source.onerror = function () {
      source.close();
      onEvent({ ev: "stream_error" });
    };
    return source;
  }

  return {
    listWorkflows: function () {
      return get("/api/workflows");
    },
    createWorkflow: function (name) {
      return post("/api/workflows", { name: name });
    },
    getWorkflow: function (name) {
      return get("/api/workflows/" + encodeURIComponent(name));
    },
    updateWorkflow: function (name, data) {
      return put("/api/workflows/" + encodeURIComponent(name), data);
    },
    deleteWorkflowRuns: function (name) {
      return del("/api/workflows/" + encodeURIComponent(name) + "/runs");
    },
    getInfra: function (name) {
      return get(
        "/api/workflows/" + encodeURIComponent(name) + "/infrastructure"
      );
    },
    setInfra: function (name, data) {
      return put(
        "/api/workflows/" + encodeURIComponent(name) + "/infrastructure",
        data
      );
    },
    runWorkflow: function (name) {
      return post("/api/workflows/" + encodeURIComponent(name) + "/run");
    },
    stopWorkflow: function (name) {
      return post("/api/workflows/" + encodeURIComponent(name) + "/stop");
    },
    getSchema: function (schemaName) {
      return get("/api/schemas/" + encodeURIComponent(schemaName));
    },
    getRecentWorkflows: function () {
      return get("/api/recent-workflows");
    },
    getRunLog: function (runId) {
      return get("/api/runs/" + encodeURIComponent(runId) + "/log");
    },
    getWorkflowRunLog: function (name, runId) {
      return get(
        "/api/workflows/" +
          encodeURIComponent(name) +
          "/runs/" +
          encodeURIComponent(runId) +
          "/log"
      );
    },
    listRuns: function (name) {
      return get("/api/workflows/" + encodeURIComponent(name) + "/runs");
    },
    openRunDir: function (name, runId) {
      return post("/api/workflows/" + encodeURIComponent(name) + "/runs/" + encodeURIComponent(runId) + "/open");
    },
    analyzeBucketStats: function (sourceDir, enabledFamilies) {
      return post("/api/dataset/bucket-stats", {
        source_dir: sourceDir,
        enabled_families: enabledFamilies,
      });
    },
    getSettings: function () {
      return get("/api/settings");
    },
    setSettings: function (data) {
      return put("/api/settings", data);
    },
    browse: function (path, type) {
      return post("/api/browse", { path: path, type: type || "directory" });
    },
    connectEventStream: connectEventStream,
  };
})();
