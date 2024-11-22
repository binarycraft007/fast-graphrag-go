package googleai

import (
	"reflect"
	"strings"

	"github.com/google/generative-ai-go/genai"
)

// GenerateSchemaFromType converts a reflect.Type to a Schema object.
func GenerateSchemaFromType(typ reflect.Type) (*genai.Schema, error) {
	schema := &genai.Schema{}

	switch typ.Kind() {
	case reflect.Struct:
		schema.Type = genai.TypeObject
		schema.Properties = make(map[string]*genai.Schema)
		schema.Required = []string{}

		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			fieldSchema, err := parseFieldSchema(field)
			if err != nil {
				return nil, err
			}

			jsonTag := field.Tag.Get("json")
			if jsonTag == "" || jsonTag == "-" {
				continue
			}

			jsonParts := strings.Split(jsonTag, ",")
			jsonName := jsonParts[0]

			if !contains(jsonParts, "omitempty") {
				schema.Required = append(schema.Required, jsonName)
			}

			schema.Properties[jsonName] = fieldSchema
		}

	case reflect.Slice, reflect.Array:
		schema.Type = genai.TypeArray
		elemType := typ.Elem()

		itemSchema, err := GenerateSchemaFromType(elemType)
		if err != nil {
			return nil, err
		}

		schema.Items = itemSchema

	default:
		schema.Type = goTypeToSchemaType(typ)
		schema.Format = goTypeToSchemaFormat(typ)
		schema.Nullable = isNullable(typ)
	}

	return schema, nil
}

// parseFieldSchema parses a single field to a Schema, handling arrays, structs, and formats.
func parseFieldSchema(field reflect.StructField) (*genai.Schema, error) {
	fieldType := field.Type
	schema := &genai.Schema{
		Type:     goTypeToSchemaType(fieldType),
		Nullable: isNullable(fieldType),
	}

	// Set format for primitive types
	schema.Format = goTypeToSchemaFormat(fieldType)

	switch fieldType.Kind() {
	case reflect.Slice, reflect.Array:
		elemType := fieldType.Elem()
		itemSchema, err := GenerateSchemaFromType(elemType)
		if err != nil {
			return nil, err
		}
		schema.Items = itemSchema

	case reflect.Struct:
		nestedSchema, err := GenerateSchemaFromType(fieldType)
		if err != nil {
			return nil, err
		}
		schema.Type = genai.TypeObject
		schema.Properties = nestedSchema.Properties
	}

	schemaTag := field.Tag.Get("schema")
	if schemaTag != "" {
		tags := parseTags(schemaTag)
		if desc, ok := tags["description"]; ok {
			schema.Description = desc
		}
		if example, ok := tags["example"]; ok {
			schema.Enum = strings.Split(example, ",")
		}
	}

	return schema, nil
}

// Helper functions

func isNullable(typ reflect.Type) bool {
	return typ.Kind() == reflect.Ptr ||
		typ.Kind() == reflect.Interface ||
		typ.Kind() == reflect.Slice ||
		typ.Kind() == reflect.Map ||
		typ.Kind() == reflect.Chan
}

func contains(slice []string, value string) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}

func parseTags(tag string) map[string]string {
	parts := strings.Split(tag, ",")
	tags := make(map[string]string)

	for _, part := range parts {
		kv := strings.SplitN(part, "=", 2)
		if len(kv) == 2 {
			tags[kv[0]] = kv[1]
		}
	}

	return tags
}

func goTypeToSchemaType(typ reflect.Type) genai.Type {
	switch typ.Kind() {
	case reflect.String:
		return genai.TypeString
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return genai.TypeInteger
	case reflect.Float32, reflect.Float64:
		return genai.TypeNumber
	case reflect.Bool:
		return genai.TypeBoolean
	case reflect.Slice, reflect.Array:
		return genai.TypeArray
	case reflect.Struct:
		return genai.TypeObject
	default:
		return genai.TypeString
	}
}

func goTypeToSchemaFormat(typ reflect.Type) string {
	switch typ.Kind() {
	case reflect.Int, reflect.Int32:
		return "int32"
	case reflect.Int64:
		return "int64"
	case reflect.Float32:
		return "float"
	case reflect.Float64:
		return "double"
	default:
		return ""
	}
}
