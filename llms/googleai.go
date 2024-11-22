package llms

import (
	"context"
	"errors"
	"reflect"

	"github.com/binarycraft007/fast-graphrag-go/llms/googleai"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type LLMGoogleAI struct {
	client *genai.Client
}

func NewLLMGoogleAI(ctx context.Context, opts ...option.ClientOption) (*LLMGoogleAI, error) {
	client, err := genai.NewClient(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &LLMGoogleAI{client: client}, nil
}

func (l *LLMGoogleAI) SendMessage(
	ctx context.Context,
	prompt string,
	model string,
	systemPrompt string,
	historyMessages interface{},
	responseType reflect.Type,
) (*genai.GenerateContentResponse, error) {
	genaiModel := l.client.GenerativeModel(model)
	genaiModel.SetCandidateCount(1)

	cs := genaiModel.StartChat()
	if history, ok := historyMessages.([]*genai.Content); ok {
		cs.History = history
	} else {
		return nil, errors.ErrUnsupported
	}
	switch responseType.Kind() {
	case reflect.String:
		return cs.SendMessage(ctx, genai.Text(prompt))
	case reflect.Slice, reflect.Array, reflect.Struct:
		schema, err := googleai.GenerateSchemaFromType(responseType)
		if err != nil {
			return nil, err
		}
		genaiModel.GenerationConfig.ResponseMIMEType = "application/json"
		genaiModel.GenerationConfig.ResponseSchema = schema
		return cs.SendMessage(ctx, genai.Text(prompt))
	default:
		return nil, errors.New("unsupported type")
	}
}
