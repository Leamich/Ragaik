<script lang="ts">
	import MarkdownWithMath from '$lib/MarkdownWithMath.svelte';
	import {askQuery, loadHistory} from '../../api';
	import type { QuerySchema, ResponseSchema } from '../../types/schema';
	import {onMount} from "svelte";

	let messages = [
		{
			id: 1,
			user: 'bot',
			text: 'Hello! I am RAGaik, your personal assistant for studying. How can I help you today?'
		}
	];

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	let testMessages = [
		{
			id: 1,
			user: 'bot',
			text: 'Hello! I am RAGaik, your personal assistant for studying. How can I help you today?'
		},
		{
			id: 2,
			user: 'user',
			text: 'Can you explain the Pythagorean theorem?'
		},
		{
			id: 3,
			user: 'bot',
			text: 'Certainly! The **Pythagorean theorem** states that for a right triangle:\n\n\\[ a^2 + b^2 = c^2 \\]\n\nwhere \\( a \\) and \\( b \\) are the legs and \\( c \\) is the hypotenuse.'
		},
		{
			id: 4,
			user: 'user',
			text: 'How do I solve for \\( x \\) in the equation \\( 2x + 3 = 7 \\)?'
		},
		{
			id: 5,
			user: 'bot',
			text: 'To solve for \\( x \\): \n\n 1. Subtract 3 from both sides: \n \\[ 2x = 4 \\] \n\n 2. Divide both sides by 2:\n \\[ x = 2 \\] \n\n'
		},
		{
			id: 6,
			user: 'user',
			text: 'What is the derivative of \\( f(x) = x^2 \\)?'
		},
		{
			id: 7,
			user: 'bot',
			text: "The derivative of \\( f(x) = x^2 \\) is:\n\n\\[ f'(x) = 2x \\]"
		}
	];

	let input = '';
	let loading = false;
	let nextId = 2;

	onMount(async () => {
		const loadedMessages = await loadHistory()
		if (loadedMessages && loadedMessages.length > 0) {
			messages = loadedMessages.map((msg, index) => ({
				id: nextId + index - 1,
				user: index % 2 === 0 ? 'user' : 'bot',
				text: msg
			}));
			nextId = messages.length + 1;
		}
	});

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
		} catch {
			messages = [
				...messages,
				{ id: nextId++, user: 'bot', text: 'Sorry, there was an error contacting the backend.' }
			];
		} finally {
			loading = false;
		}
	}
</script>

<div class="relative max-h-fit min-h-full bg-white">
	<!-- Chat messages with typography styling -->
	<div
		class="prose prose-blue max-w-full max-h-[calc(100vh-80px)] min-h-[calc(100vh-80px)] px-4 py-6 overflow-y-scroll h-full chat-container pb-32"
	>
		{#each messages as msg (msg.id)}
			<div
				id="msg-{msg.id}"
				class="mb-4 p-3 rounded-lg shadow-sm"
				class:bg-blue-100={msg.user === 'bot'}
				class:bg-blue-200={msg.user === 'user'}
			>
				<!-- eslint-disable-next-line svelte/no-at-html-tags -->
				<MarkdownWithMath text={msg.text} />
			</div>
		{/each}
		{#if loading}
			<div class="mb-4 p-3 rounded-lg shadow-sm bg-blue-100 opacity-70">
				<p>Bot is typing…</p>
			</div>
		{/if}
	</div>

	<!-- Prompt overlay -->
	<div
		class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-blue-100 via-blue-50 to-transparent bg-opacity-95 backdrop-blur-lg p-4 shadow-lg border-t border-blue-200"
	>
		<form class="flex items-center gap-3" on:submit|preventDefault={handleSubmit}>
			<div class="flex-1 relative">
				<textarea
					bind:value={input}
					rows="1"
					placeholder="Type your message..."
					class="w-full p-3 h-12 pr-12 border-2 border-blue-200 rounded-2xl shadow focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 bg-white/80 transition resize-none text-base placeholder-gray-400"
					disabled={loading}
					style="min-height: 44px; max-height: 120px;"
				></textarea>
				<!-- Optional: Add a subtle icon inside the input (e.g., a pencil) -->
				<svg
					class="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-blue-300 pointer-events-none"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					viewBox="0 0 24 24"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M15.232 5.232l3.536 3.536M9 13l6.768-6.768a2 2 0 112.828 2.828L11.828 15.828a4 4 0 01-1.414.94l-4.243 1.415 1.415-4.243a4 4 0 01.94-1.414z"
					/>
				</svg>
			</div>
			<button
				type="submit"
				class="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold rounded-2xl shadow hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300 transition disabled:opacity-60 disabled:cursor-not-allowed"
				disabled={loading || !input.trim()}
			>
				{#if loading}
					<svg class="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"
						></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
						></path>
					</svg>
					Sending...
				{:else}
					<span>Send</span>
					<svg
						class="w-5 h-5 ml-1"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						viewBox="0 0 24 24"
					>
						<path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14M12 5l7 7-7 7" />
					</svg>
				{/if}
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
