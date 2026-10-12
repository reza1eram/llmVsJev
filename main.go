// Jev vs LLM race - single-file Windows build of server.py.
// Build: GOOS=windows GOARCH=amd64 go build -o llmVsJev.exe .
package main

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

//go:embed index.html
var indexHTML []byte

//go:embed fonts/*.woff2
var fontsFS embed.FS

type Step struct {
	ID, Title, Instructions, True, False string
}

var steps = []Step{
	{"is_bug", "آیا باگ نرم‌افزاری است؟", "Is the customer reporting a software defect?",
		"The customer describes broken or unexpected product behavior.",
		"The customer is asking a question or requesting a feature."},
	{"is_urgent", "آیا فوری است؟", "Does this ticket need urgent handling?",
		"The issue blocks payment, causes data or money loss, a security risk, or the customer cannot use the product at all.",
		"The issue is minor, cosmetic, a general question, or can wait for normal handling."},
	{"is_angry", "آیا مشتری ناراحت است؟", "Is the customer frustrated, angry, or threatening to leave?",
		"The message shows frustration, anger, complaints about repeated problems, or threats to cancel.",
		"The message is calm, neutral, or polite."},
	{"wants_refund", "آیا درخواست بازگشت پول دارد؟", "Is the customer asking for a refund, chargeback, or compensation?",
		"The customer explicitly or implicitly asks to get money back or be compensated.",
		"The customer does not ask for any money back or compensation."},
	{"needs_human", "ارجاع به کارشناس انسانی؟", "Should this ticket be escalated to a senior human agent instead of an automated reply?",
		"The case involves money, legal or security risk, an upset customer with an urgent problem, or something an automated reply cannot solve.",
		"A clear automated reply with instructions is enough."},
}

var cfg = map[string]string{
	"JEV_BASE_URL": "https://kayaai.ir/api",
	"JEV_MODEL":    "typesafe/jev-1.13",
	"LLM_BASE_URL": "https://api.avalai.ir/v1",
	"LLM_MODEL":    "gpt-4.1-mini",
	"PORT":         "8000",
}

func loadEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") || !strings.Contains(line, "=") {
			continue
		}
		kv := strings.SplitN(line, "=", 2)
		cfg[strings.TrimSpace(kv[0])] = strings.Trim(strings.TrimSpace(kv[1]), `"'`)
	}
}

func conf(k string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return cfg[k]
}

var client = &http.Client{Timeout: 90 * time.Second}

func postJSON(url, key string, payload any) (map[string]any, float64, error) {
	body, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	t0 := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	ms := float64(time.Since(t0).Microseconds()) / 1000
	var data map[string]any
	if json.Unmarshal(raw, &data) != nil {
		data = map[string]any{"raw_text": string(raw)}
	}
	if resp.StatusCode >= 400 {
		s := string(raw)
		if len(s) > 500 {
			s = s[:500]
		}
		return nil, ms, fmt.Errorf("HTTP %d from %s: %s", resp.StatusCode, url, s)
	}
	return data, ms, nil
}

// toBool normalises whatever shape an answer comes back in to true/false/nil.
func toBool(v any) any {
	switch x := v.(type) {
	case nil:
		return nil
	case bool:
		return x
	case float64:
		return x != 0
	case map[string]any:
		if p, ok := x["noul"].(float64); ok { // Jev returns a probability
			return p >= 0.5
		}
		for _, k := range []string{"value", "answer", "result", "decision"} {
			if vv, ok := x[k]; ok {
				return toBool(vv)
			}
		}
		return nil
	case string:
		switch strings.ToLower(strings.TrimSpace(x)) {
		case "true", "yes", "1", "y":
			return true
		case "false", "no", "0", "n":
			return false
		}
	}
	return nil
}

func num(m map[string]any, keys ...string) float64 {
	for _, k := range keys {
		if f, ok := m[k].(float64); ok {
			return f
		}
	}
	return 0
}

func usageOf(data map[string]any) any {
	u, ok := data["usage"].(map[string]any)
	if !ok {
		return nil
	}
	p := num(u, "prompt_tokens", "input_tokens")
	c := num(u, "completion_tokens", "output_tokens")
	t := num(u, "total_tokens")
	if t == 0 {
		t = p + c
	}
	return map[string]any{"prompt": p, "completion": c, "total": t}
}

func jevDecide(ticket string, s Step) (map[string]any, error) {
	payload := map[string]any{
		"model": conf("JEV_MODEL"),
		"state": map[string]any{"ticket": ticket},
		"questions": map[string]any{s.ID: map[string]any{
			"type": "noul", "instructions": s.Instructions,
			"criteria": map[string]any{"true": s.True, "false": s.False},
		}},
	}
	data, ms, err := postJSON(strings.TrimRight(conf("JEV_BASE_URL"), "/")+"/decisions", conf("JEV_API_KEY"), payload)
	if err != nil {
		return nil, err
	}
	answers, ok := data["answers"].(map[string]any)
	if !ok {
		answers = data
	}
	var confidence any
	if m, ok := answers[s.ID].(map[string]any); ok {
		confidence = m["noul"]
	}
	return map[string]any{"decision": toBool(answers[s.ID]), "confidence": confidence, "reasoning": nil, "ms": ms,
		"usage": usageOf(data), "request": payload, "response": data}, nil
}

// jevBatch asks all five decisions in a single Jev request.
func jevBatch(ticket string) (map[string]any, error) {
	questions := map[string]any{}
	for _, s := range steps {
		questions[s.ID] = map[string]any{"type": "noul", "instructions": s.Instructions,
			"criteria": map[string]any{"true": s.True, "false": s.False}}
	}
	payload := map[string]any{"model": conf("JEV_MODEL"), "state": map[string]any{"ticket": ticket}, "questions": questions}
	data, ms, err := postJSON(strings.TrimRight(conf("JEV_BASE_URL"), "/")+"/decisions", conf("JEV_API_KEY"), payload)
	if err != nil {
		return nil, err
	}
	answers, _ := data["answers"].(map[string]any)
	out := map[string]any{}
	for _, s := range steps {
		var confidence any
		if m, ok := answers[s.ID].(map[string]any); ok {
			confidence = m["noul"]
		}
		out[s.ID] = map[string]any{"decision": toBool(answers[s.ID]), "confidence": confidence}
	}
	return map[string]any{"steps": out, "ms": ms, "usage": usageOf(data), "reasoning": nil,
		"request": payload, "response": data}, nil
}

func llmChat(messages []map[string]string, maxTokens int, temp float64) (string, float64, any, any, map[string]any, error) {
	payload := map[string]any{"model": conf("LLM_MODEL"), "messages": messages, "temperature": temp, "max_tokens": maxTokens}
	data, ms, err := postJSON(strings.TrimRight(conf("LLM_BASE_URL"), "/")+"/chat/completions", conf("LLM_API_KEY"), payload)
	if err != nil {
		return "", ms, nil, payload, nil, err
	}
	text := ""
	if ch, ok := data["choices"].([]any); ok && len(ch) > 0 {
		if m, ok := ch[0].(map[string]any)["message"].(map[string]any); ok {
			text, _ = m["content"].(string)
		}
	}
	return text, ms, usageOf(data), payload, data, nil
}

var jsonObj = regexp.MustCompile(`(?s)\{.*\}`)

func llmDecide(ticket string, s Step) (map[string]any, error) {
	system := "You are a support-ticket triage engine. Think step by step about the ticket, " +
		"then give a final yes/no decision. Reply ONLY with JSON: " +
		`{"reasoning": "<short reasoning>", "answer": true|false}`
	user := fmt.Sprintf("Ticket:\n%s\n\nQuestion: %s\nAnswer true if: %s\nAnswer false if: %s", ticket, s.Instructions, s.True, s.False)
	text, ms, usage, payload, data, err := llmChat([]map[string]string{{"role": "system", "content": system}, {"role": "user", "content": user}}, 400, 0)
	if err != nil {
		return nil, err
	}
	var decision any
	reasoning := any(text)
	if m := jsonObj.FindString(text); m != "" {
		var j map[string]any
		if json.Unmarshal([]byte(m), &j) == nil {
			decision = toBool(j["answer"])
			if r, ok := j["reasoning"]; ok {
				reasoning = r
			}
		}
	}
	return map[string]any{"decision": decision, "reasoning": reasoning, "ms": ms,
		"usage": usage, "request": payload, "response": data}, nil
}

func routeTeam(d map[string]any) string {
	if d["wants_refund"] == true {
		return "Billing"
	}
	if d["is_bug"] == true {
		return "Engineering"
	}
	return "Customer Success"
}

func writeReply(ticket string, decisions map[string]any) (map[string]any, error) {
	team := routeTeam(decisions)
	var facts []string
	for _, s := range steps {
		facts = append(facts, fmt.Sprintf("- %s: %v", s.ID, decisions[s.ID]))
	}
	system := "You write short, friendly customer-support replies (max 80 words) " +
		"in the same language as the ticket. Follow the triage facts exactly."
	user := fmt.Sprintf("Ticket:\n%s\n\nTriage facts:\n%s\n- routed_team: %s\n\nWrite the reply.", ticket, strings.Join(facts, "\n"), team)
	text, ms, usage, payload, data, err := llmChat([]map[string]string{{"role": "system", "content": system}, {"role": "user", "content": user}}, 250, 0.3)
	if err != nil {
		return nil, err
	}
	return map[string]any{"text": text, "ms": ms, "usage": usage, "request": payload, "response": data, "team": team}, nil
}

func send(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
}

func main() {
	exe, _ := os.Executable()
	dir := filepath.Dir(exe)
	for _, p := range []string{filepath.Join(dir, "config.env"), filepath.Join(dir, ".env"), "config.env", ".env"} {
		loadEnv(p)
	}
	keysOK := conf("JEV_API_KEY") != "" && conf("LLM_API_KEY") != ""

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" && r.URL.Path != "/index.html" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	http.Handle("/fonts/", http.FileServer(http.FS(fontsFS)))
	http.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		var list []map[string]string
		for _, s := range steps {
			list = append(list, map[string]string{"id": s.ID, "title": s.Title})
		}
		send(w, 200, map[string]any{"steps": list, "jev_model": conf("JEV_MODEL"), "llm_model": conf("LLM_MODEL"), "keys_ok": keysOK})
	})
	http.HandleFunc("/api/step", func(w http.ResponseWriter, r *http.Request) {
		var b struct{ Path, Step, Ticket string }
		json.NewDecoder(r.Body).Decode(&b)
		for _, s := range steps {
			if s.ID == b.Step {
				var out map[string]any
				var err error
				if b.Path == "jev" {
					out, err = jevDecide(b.Ticket, s)
				} else {
					out, err = llmDecide(b.Ticket, s)
				}
				if err != nil {
					send(w, 500, map[string]string{"error": err.Error()})
				} else {
					send(w, 200, out)
				}
				return
			}
		}
		send(w, 404, map[string]string{"error": "unknown step"})
	})
	http.HandleFunc("/api/jev_batch", func(w http.ResponseWriter, r *http.Request) {
		var b struct{ Ticket string }
		json.NewDecoder(r.Body).Decode(&b)
		out, err := jevBatch(b.Ticket)
		if err != nil {
			send(w, 500, map[string]string{"error": err.Error()})
			return
		}
		send(w, 200, out)
	})
	http.HandleFunc("/api/reply", func(w http.ResponseWriter, r *http.Request) {
		var b struct {
			Ticket    string         `json:"ticket"`
			Decisions map[string]any `json:"decisions"`
		}
		json.NewDecoder(r.Body).Decode(&b)
		out, err := writeReply(b.Ticket, b.Decisions)
		if err != nil {
			send(w, 500, map[string]string{"error": err.Error()})
			return
		}
		send(w, 200, out)
	})

	// use the configured port, or the next free one if it is taken
	var ln net.Listener
	var err error
	port := 0
	fmt.Sscan(conf("PORT"), &port)
	for i := 0; i < 20; i++ {
		if ln, err = net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port+i)); err == nil {
			port += i
			break
		}
	}
	if err != nil {
		fmt.Println("Could not open a port:", err)
		fmt.Scanln()
		return
	}
	url := fmt.Sprintf("http://localhost:%d", port)
	fmt.Println("Jev vs LLM race")
	fmt.Printf("Jev model: %s | LLM model: %s\n", conf("JEV_MODEL"), conf("LLM_MODEL"))
	if !keysOK {
		fmt.Println("WARNING: JEV_API_KEY / LLM_API_KEY missing. Put config.env next to the exe.")
	}
	fmt.Println("Running at", url, " (close this window to stop)")
	if os.Getenv("NO_BROWSER") == "" {
		openBrowser(url)
	}
	http.Serve(ln, nil)
}

func openBrowser(url string) {
	switch runtime.GOOS {
	case "windows":
		exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		exec.Command("open", url).Start()
	default:
		exec.Command("xdg-open", url).Start()
	}
}
