local envs = {
  Remark   = "Remark",
  Example  = "Example",
  defin    = "Definition",
  lem      = "Lemma",
  theo     = "Theorem",
  nota     = "Notation",
  propos   = "Proposition", 
  Exercise = "Exercise",
  Proof    = "Proof"
}

local function inlines_to_string(inlines)
  return pandoc.utils.stringify(inlines)
end

function Div(el)
  local cls = el.classes and el.classes[1]
  if cls and envs[cls] then
    local fullName = envs[cls]
    local blocks = el.content
    local titleArg = nil

    if #blocks > 0 and blocks[1].t == 'Para' then
      local inls = blocks[1].content
      if #inls > 0 and inls[1].t == 'Span' and #inls[1].classes == 0 then
        titleArg = inlines_to_string(inls[1].content)
        table.remove(inls, 1)
        if #inls > 0 and inls[1].t == 'SoftBreak' then
          table.remove(inls, 1)
        end
        if #inls == 0 then
          table.remove(blocks, 1)
        else
          blocks[1].content = inls
        end
      end
    end

    local headerText = fullName
    if titleArg and titleArg ~= '' then
      headerText = headerText .. ". " .. titleArg .. "."
    end
    local header = pandoc.Header(2, headerText)
    return { header, table.unpack(blocks) }
  end
  return nil
end
