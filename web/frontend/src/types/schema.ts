interface QuerySchema {
	query: string;
}

interface MessageResponseSchema {
	text: string;
	image_ids?: string[];
}

export type { QuerySchema, MessageResponseSchema };
