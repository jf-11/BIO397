function area(circle::Circle2)
    return π * circle.radius^2
end

function area(square::Square)
    return square.side^2
end

function overlapping(circ1::Circle2,circ2::Circle2)
    distance = sqrt((circ1.point.x - circ2.point.x)^2 + (circ1.point.y - circ2.point.y)^2)
    if distance < circ1.radius + circ2.radius
        return "The circles do overlap each other."
    else
        return "The circles do not overlap each other."
    end
end