-- Presentation only: preserve every caption, cell, number and scientific paragraph.
function Meta(meta)
  meta.subtitle = meta['draft-status']
  meta.date = pandoc.MetaString('')
  return meta
end

function Header(header)
  if header.level > 1 then header.level = header.level - 1 end
  if pandoc.utils.stringify(header.content) == 'References' then
    return {pandoc.RawBlock('latex', '\\clearpage'), header}
  end
  return header
end

function Pandoc(doc)
  -- Author details come from canonical metadata and follow the generated title.
  local details = doc.meta['author-details']
  if details then
    doc.blocks:insert(1, pandoc.RawBlock('latex', '\\end{center}'))
    for i = #details, 1, -1 do doc.blocks:insert(1, details[i]) end
    doc.blocks:insert(1, pandoc.RawBlock('latex', '\\begin{center}\\small'))
  end
  return doc
end

function Link(link)
  if not link.target:match('^%a[%w+.-]*:') and not link.target:match('^#') then
    local target = link.target:gsub('^%.%./', '')
    link.target = os.getenv('PAPER_SUPPORT_URL') .. '/' .. target
  end
  return link
end

function Blocks(blocks)
  local result = pandoc.List()
  for _, block in ipairs(blocks) do
    if block.t == 'Table' then
      local columns = #block.colspecs
      local caption = nil
      if #result > 0 and result[#result].t == 'Para'
          and pandoc.utils.stringify(result[#result]):match('^Table %d+%.') then
        caption = result:remove()
      end
      local wide = columns >= 6
      if columns == 5 then
        for i = 1, columns do
          block.colspecs[i][2] = i == 1 and 0.32 or 0.17
        end
      elseif columns == 4 then
        for i = 1, columns do
          block.colspecs[i][2] = i == 1 and 0.19 or (i == 4 and 0.35 or 0.23)
        end
      elseif columns == 3 then
        block.colspecs[1][2] = 0.42
        block.colspecs[2][2] = 0.29
        block.colspecs[3][2] = 0.29
      elseif wide then
        local label = caption and pandoc.utils.stringify(caption) or ''
        local widths
        if label:match('^Table 3%.') then
          widths = {0.14, 0.08, 0.16, 0.23, 0.21, 0.18}
        elseif label:match('^Table 6%.') then
          widths = {0.18, 0.135, 0.135, 0.135, 0.135, 0.28}
        else
          widths = {0.14, 0.145, 0.145, 0.145, 0.145, 0.28}
        end
        for i = 1, columns do block.colspecs[i][2] = widths[i] end
      end
      local label = caption and pandoc.utils.stringify(caption) or ''
      local needed = label:match('^Table 2%.') and 16 or 10
      result:insert(pandoc.RawBlock('latex', '\\begingroup\\Needspace{' .. needed .. '\\baselineskip}'))
      if caption then result:insert(caption) end
      result:insert(block)
      result:insert(pandoc.RawBlock('latex', '\\endgroup'))
    else
      result:insert(block)
    end
  end
  return result
end
