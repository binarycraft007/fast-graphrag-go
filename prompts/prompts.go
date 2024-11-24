package prompts

import (
	_ "embed"
)

//go:embed entity_relationship_extraction.md
var entity_relationship_extraction string

//go:embed entity_relationship_extraction.json
var EntityRelationshipExtractionExample string

var Prompts = map[string]string{
	"entity_relationship_extraction":               entity_relationship_extraction,
	"entity_relationship_continue_extraction":      "MANY entities were missed in the last extraction.  Add them below using the same format:",
	"entity_relationship_gleaning_done_extraction": "Retrospectively check if all entities have been correctly identified: answer done if so, or continue if there are still entities that need to be added.",
}
