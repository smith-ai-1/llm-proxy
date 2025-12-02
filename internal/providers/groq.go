package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/gorilla/mux"
)

const groqBaseURL = "https://api.groq.com"

// GroqProxy implements an OpenAI-compatible proxy targeting Groq's API
type GroqProxy struct {
	proxy  *httputil.ReverseProxy
	parser *OpenAIProxy
}

// NewGroqProxy creates a Groq reverse proxy
func NewGroqProxy() *GroqProxy {
	targetURL, err := url.Parse(groqBaseURL)
	if err != nil {
		log.Fatalf("Failed to parse Groq API URL: %v", err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	groqProxy := &GroqProxy{
		proxy:  proxy,
		parser: &OpenAIProxy{},
	}

	originalDirector := proxy.Director
	baseDirector := CreateGenericDirector(groqProxy, targetURL, originalDirector)
	proxy.Director = func(req *http.Request) {
		baseDirector(req)
		if !strings.HasPrefix(req.URL.Path, "/openai/") {
			req.URL.Path = "/openai" + req.URL.Path
		}
	}
	proxy.Transport = newProxyTransport()

	proxy.ModifyResponse = func(resp *http.Response) error {
		if groqProxy.isStreamingResponse(resp) {
			log.Printf("Detected streaming response from Groq")

			resp.Header.Set("Cache-Control", "no-cache")
			resp.Header.Set("Connection", "keep-alive")
			resp.Header.Set("X-Accel-Buffering", "no")

			resp.Header.Del("Content-Length")
		}
		return nil
	}

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Groq proxy error: %v", err)

		if groqProxy.IsStreamingRequest(r) {
			if w.Header().Get("Content-Type") == "" {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.WriteHeader(http.StatusBadGateway)
				fmt.Fprintf(w, "data: {\"error\": \"Proxy error: %v\"}\n\n", err)
				fmt.Fprintf(w, "data: [DONE]\n\n")
			} else {
				log.Printf("Cannot send error response, headers already sent")
			}
		} else {
			w.WriteHeader(http.StatusBadGateway)
			fmt.Fprintf(w, "Groq proxy error: %v", err)
		}
	}

	return groqProxy
}

// GetName returns provider name
func (g *GroqProxy) GetName() string {
	return "groq"
}

// IsStreamingRequest detects streaming intents based on headers and body
func (g *GroqProxy) IsStreamingRequest(req *http.Request) bool {
	if strings.Contains(req.Header.Get("Accept"), "text/event-stream") {
		return true
	}

	if !strings.HasPrefix(req.URL.Path, "/groq/") {
		return false
	}

	if req.Method == "POST" && (strings.Contains(req.URL.Path, "/chat/completions") ||
		strings.Contains(req.URL.Path, "/completions") ||
		strings.Contains(req.URL.Path, "/responses")) {
		return g.parser.checkStreamingInBody(req)
	}

	return false
}

// ParseResponseMetadata reuses OpenAI parsing and fixes provider name
func (g *GroqProxy) ParseResponseMetadata(responseBody io.Reader, isStreaming bool) (*LLMResponseMetadata, error) {
	metadata, err := g.parser.ParseResponseMetadata(responseBody, isStreaming)
	if metadata != nil {
		metadata.Provider = g.GetName()
	}
	return metadata, err
}

// Proxy returns underlying reverse proxy
func (g *GroqProxy) Proxy() http.Handler {
	return g.proxy
}

// GetHealthStatus returns readiness info
func (g *GroqProxy) GetHealthStatus() map[string]interface{} {
	return map[string]interface{}{
		"provider":          g.GetName(),
		"status":            "healthy",
		"baseURL":           groqBaseURL,
		"streaming_support": true,
		"body_parsing":      true,
	}
}

// UserIDFromRequest extracts Groq `user` fields
func (g *GroqProxy) UserIDFromRequest(req *http.Request) string {
	if req.Body == nil || req.Method != "POST" {
		return ""
	}

	if !strings.HasPrefix(req.URL.Path, "/groq/") {
		return ""
	}

	bodyBytes, err := g.parser.readRequestBodyForUserID(req)
	if err != nil {
		log.Printf("Error reading Groq request body for user ID extraction: %v", err)
		return ""
	}

	if len(bodyBytes) == 0 {
		return ""
	}

	var data map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		log.Printf("Error parsing Groq request JSON for user ID extraction: %v", err)
		return ""
	}

	if userValue, ok := data["user"].(string); ok && userValue != "" {
		log.Printf("🔍 Groq: Extracted user ID: %s", userValue)
		return userValue
	}

	return ""
}

// RegisterExtraRoutes no-op for Groq
func (g *GroqProxy) RegisterExtraRoutes(router *mux.Router) {}

// ValidateAPIKey handles iw: mapping similar to OpenAI
func (g *GroqProxy) ValidateAPIKey(req *http.Request, keyStore APIKeyStore) error {
	authHeader := req.Header.Get("Authorization")
	if authHeader == "" {
		return nil
	}

	const bearerPrefix = "Bearer "
	if !strings.HasPrefix(authHeader, bearerPrefix) {
		return nil
	}

	apiKey := strings.TrimPrefix(authHeader, bearerPrefix)

	actualKey, provider, err := keyStore.ValidateAndGetActualKey(context.Background(), apiKey)
	if err != nil {
		return fmt.Errorf("API key validation failed: %w", err)
	}

	if provider != "" && provider != g.GetName() {
		return fmt.Errorf("API key is for provider %s, not %s", provider, g.GetName())
	}

	if actualKey != apiKey {
		req.Header.Set("Authorization", bearerPrefix+actualKey)
		log.Printf("🔑 Groq: Translated API key from iw: format")
	}

	return nil
}

// ExtractRequestModelAndMessages pulls model/message text from Groq requests
func (g *GroqProxy) ExtractRequestModelAndMessages(req *http.Request) (string, []string) {
	if req == nil || req.Method != "POST" || !strings.HasPrefix(req.URL.Path, "/groq/") {
		return "", nil
	}

	bodyBytes, err := g.parser.readRequestBodyForUserID(req)
	if err != nil || len(bodyBytes) == 0 {
		return "", nil
	}

	var data map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return "", nil
	}

	model := ""
	if mv, ok := data["model"].(string); ok {
		model = mv
	}

	messages := make([]string, 0, 8)

	if rawMsgs, ok := data["messages"].([]interface{}); ok {
		for _, m := range rawMsgs {
			if msg, ok := m.(map[string]interface{}); ok {
				if contentStr, ok := msg["content"].(string); ok && contentStr != "" {
					messages = append(messages, contentStr)
					continue
				}
				if parts, ok := msg["content"].([]interface{}); ok {
					for _, p := range parts {
						if pm, ok := p.(map[string]interface{}); ok {
							if t, ok := pm["type"].(string); ok && t == "text" {
								if txt, ok := pm["text"].(string); ok && txt != "" {
									messages = append(messages, txt)
								}
							}
						}
					}
				}
			}
		}
	}

	if inputStr, ok := data["input"].(string); ok && inputStr != "" {
		messages = append(messages, inputStr)
	} else if inputArr, ok := data["input"].([]interface{}); ok {
		for _, it := range inputArr {
			if m, ok := it.(map[string]interface{}); ok {
				if t, ok := m["type"].(string); ok && (t == "input_text" || t == "text") {
					if txt, ok := m["text"].(string); ok && txt != "" {
						messages = append(messages, txt)
					}
				}
			} else if s, ok := it.(string); ok && s != "" {
				messages = append(messages, s)
			}
		}
	}

	if prompt, ok := data["prompt"].(string); ok && prompt != "" {
		messages = append(messages, prompt)
	}

	return model, messages
}

func (g *GroqProxy) isStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream")
}
