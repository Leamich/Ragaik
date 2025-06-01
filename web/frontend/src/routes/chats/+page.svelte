<script lang="ts">
	import MarkdownWithMath from '$lib/MarkdownWithMath.svelte';
	import { askQuery } from '../../api';
	import type { QuerySchema, ResponseSchema } from '../../types/schema';

	let messages = [
		{
			id: 1,
			user: 'bot',
			text: 'Hello! I am RAGaik, your personal assistant for studying. How can I help you today?'
		}
	];

	let input = '';
	let loading = false;
	let nextId = 2;

	async function handleSubmit(event: Event) {
		event.preventDefault();
		const question = input.trim();
		if (!question) return;

		// Add user message
		messages = [...messages, { id: nextId++, user: 'user', text: question }];

		input = '';
		loading = true;

		try {
			const payload: QuerySchema = { query: question };
			const res: ResponseSchema = await askQuery(payload);

			messages = [...messages, { id: nextId++, user: 'bot', text: res.response }];
		} catch (err) {
			messages = [
				...messages,
				{ id: nextId++, user: 'bot', text: 'Sorry, there was an error contacting the backend.' }
			];
		} finally {
			loading = false;
		}
	}
</script>

<div class="relative h-screen bg-gray-50">
	<!-- Chat messages with typography styling -->
	<div class="prose prose-slate max-w-full px-4 py-6 overflow-y-auto h-full chat-container pb-32">
		{#each messages as msg (msg.id)}
			<div
				id="msg-{msg.id}"
				class="mb-4 p-3 rounded-lg shadow-sm"
				class:bg-gray-100={msg.user === 'bot'}
				class:bg-blue-50={msg.user === 'user'}
			>
				<!-- eslint-disable-next-line svelte/no-at-html-tags -->
				<MarkdownWithMath text={msg.text} />
			</div>
		{/each}
		{#if loading}
			<div class="mb-4 p-3 rounded-lg shadow-sm bg-gray-100 opacity-70">
				<p>Bot is typing…</p>
			</div>
		{/if}
	</div>

	<!-- Prompt overlay -->
	<div class="absolute bottom-0 left-0 right-0 bg-white bg-opacity-90 backdrop-blur-md p-4">
		<form class="flex items-center space-x-2" on:submit|preventDefault={handleSubmit}>
			<textarea
				bind:value={input}
				rows="1"
				placeholder="Type your message..."
				class="flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring focus:ring-indigo-200 resize-none"
				disabled={loading}
			></textarea>
			<button
				type="submit"
				class="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring focus:ring-indigo-200"
				disabled={loading || !input.trim()}
			>
				{loading ? 'Sending...' : 'Send'}
			</button>
		</form>
	</div>
</div>

<style>
	/* Ensure the messages container scrolls beneath the overlay prompt */
	.chat-container {
		scrollbar-width: thin;
	}
	.chat-container::-webkit-scrollbar {
		width: 0.5rem;
	}
	.chat-container::-webkit-scrollbar-thumb {
		background-color: rgba(0, 0, 0, 0.2);
		border-radius: 0.25rem;
	}
</style>
